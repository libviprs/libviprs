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

/// The dependency resolves from crates.io, not from a git fork.
///
/// The fork existed to carry per-call locking upstream had deleted. Upstream
/// reinstated it, and what the fork still carries over the registry crate is
/// nothing libviprs calls: `set_auto_apply_intrinsic_rotation` is reverted out
/// at the pinned rev, `render_window_into_bitmap` has no call site here, and
/// both of its safety fixes are duplicated by libviprs' own guards
/// (`pdfium_page_index`, `pdfium_bitmap_span`).
///
/// The reason this is a guard rather than a preference: a git source is a
/// second dependency wearing the same name. Whoever builds from git gets the
/// fork, whoever installs from crates.io gets the registry crate, and only one
/// of them is ever tested. That divergence is what libviprs#149 was about and
/// what #981 inherited.
#[test]
#[cfg_attr(miri, ignore)] // reads Cargo.toml, and Miri isolates the filesystem
fn pdfium_render_comes_from_the_registry() {
    let decl = declaration();
    assert!(
        !decl.contains("git ="),
        "pdfium-render is declared with a git source, so a crates.io consumer \
         and a git consumer get different code under one name. The declaration \
         reads: {decl}"
    );
    assert!(
        !decl.contains("rev ="),
        "pdfium-render is pinned to a git rev: {decl}"
    );
}

/// The floor admits only versions whose `thread_safe` actually serialises.
///
/// This is the only thing protecting a published consumer. `^0.9` lets them
/// resolve 0.9.0, and nothing in that range is yanked. A lockfile, a
/// `--precise`, a vendor bundle or a sibling crate requiring `<0.9.4` all land
/// there while the manifest says it is fine.
#[test]
#[cfg_attr(miri, ignore)] // reads Cargo.toml, and Miri isolates the filesystem
fn the_floor_excludes_every_release_with_no_serialisation() {
    let decl = declaration();
    let version = decl
        .split("version = \"")
        .nth(1)
        .and_then(|s| s.split('"').next())
        .expect("a version requirement");
    let floor = version.trim_start_matches(['^', '=', '>', ' ']);
    let parts: Vec<u64> = floor.split('.').map(|p| p.parse().unwrap_or(0)).collect();
    assert!(
        parts.len() >= 3,
        "the requirement is {version:?}, which has no patch component, so it \
         admits 0.9.0 through 0.9.3. Those ship a `thread_safe` that gates \
         `unsafe impl Send + Sync` with nothing behind it."
    );
    assert!(
        (parts[0], parts[1], parts[2]) >= (0, 9, 4),
        "the requirement is {version:?} and the floor has to be at least 0.9.4"
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
    let named = decl.contains("pdfium_7881");
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
