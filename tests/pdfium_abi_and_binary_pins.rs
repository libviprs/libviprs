//! The bindgen ABI this crate requests and the libpdfium build it is run
//! against have to be the same PDFium, or the gap has to be written down
//! (issue #1017).
//!
//! `Cargo.toml` asks `pdfium-render` for `pdfium_7881`, which selects a whole
//! generated binding set. Every place that installs a libpdfium installs
//! **8054**. Those are two different PDFium milestones and until this guard
//! there was nothing in the repository that held both numbers at once, so
//! neither could be seen to disagree with the other.
//!
//! That shape is worse than an ordinary stale pin. `pdfium-render` is bound
//! dynamically here (`default-features = false`, no `static`), so a wrong
//! binding set is not a link error at build time. It is a `dlsym` of a symbol
//! that may not be there, or worse, a call through a signature the library no
//! longer has, which is a silently corrupt argument frame rather than a crash
//! you can read.
//!
//! # Why the numbers still differ, and why that is not a bug to fix here
//!
//! `pdfium-render 0.9.4` cannot bind 8054. `pdfium_7881` is the newest ABI it
//! ships and its own `pdfium_latest` is an alias for it, so requesting
//! anything newer is not an option that exists. The choice is to pin every
//! libpdfium back to 7881 or to run the gap knowingly, and the binaries moved
//! to 8054 deliberately (libviprs-tests#208, and this repo's image in #1012).
//! So the gap is real, it is currently the correct state, and what was missing
//! is anything that notices when either half moves.
//!
//! # What makes the gap safe today, measured rather than assumed
//!
//! Both halves were checked against the actual artefacts, on `linux/arm64`,
//! against the two `libviprs-dep` release tarballs the pins name. The 8054
//! tarball's digest was confirmed against the one `tools/Dockerfile.ci`
//! verifies, so this is the same binary CI installs and not a lookalike:
//!
//! * **Symbols.** `libpdfium.so` 7881 exports 635 `FUNC` symbols and 8054
//!   exports 643. 8054 is a strict superset: nothing 7881 exported was
//!   removed, and eight were added (`FORM_GetTextDirection`,
//!   `FPDFBookmark_GetColor`, `FPDFPath_GetBezierControlPoints` and five
//!   others). Every symbol `pdfium_7881.rs` declares therefore resolves.
//!
//!   Seven of the 465 declared externs resolve in *neither* binary
//!   (`FPDF_RenderPageSkia`, `FPDF_BStr_*`, `FPDF_GetRecommendedV8Flags` and
//!   the rest). They are the Skia, XFA and V8 entry points, absent from both
//!   standard builds, and `pdfium-render` only binds them behind
//!   `pdfium_use_skia` / `pdfium_enable_xfa` / `pdfium_enable_v8`, none of
//!   which this crate enables. Reading that as a 7881-versus-8054 finding
//!   would be wrong, and the 7881 control is what says so: the number is
//!   identical on both sides, so it is a property of the build flags and not
//!   of the milestone.
//!
//! * **Signatures and layouts.** Both tarballs ship PDFium's own 22 public
//!   headers, so this is a real comparison and not an inference. Exactly one
//!   declaration differs between them, and it is a struct rather than a
//!   function: `FPDF_LIBRARY_CONFIG` gained `m_BrotliEnabled` (version 6) and
//!   `m_IsolatePerDocument` (version 7). Nothing the bindings declare was
//!   removed, and no surviving function changed its parameters or return type.
//!
//!   A struct that grew two trailing fields is the one thing here that could
//!   corrupt rather than fail, because the Rust side allocates the shorter
//!   7881 layout and hands the library a pointer to it. It does not, because
//!   PDFium reads that struct by its leading `version` field and
//!   `pdfium-render 0.9.4` sets `version: 2` (`src/config.rs:224`). The new
//!   fields are gated at versions 6 and 7, so an 8054 library told it is
//!   looking at a version 2 config never reads past `m_v8EmbedderSlot`.
//!
//! So the gap is safe for this pair, for reasons that are specific to this
//! pair. Both of them stop holding the moment either number moves, which is
//! what [`DECLARED_GAP`] is for.
//!
//! # Why this reads files rather than the build
//!
//! Nothing in this repository's own job list loads libpdfium: `ci.yml` runs
//! `cargo clippy --features pdfium` and `cargo check --features pdfium` and no
//! `cargo test --features pdfium`. A runtime check would therefore not run
//! here, and PDFium's public API exposes no version query to make one out of
//! anyway. The pins are text, so the guard is over text, and it runs on every
//! job that runs the test suite.

use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf()
}

fn read(rel: &str) -> String {
    let path = repo_root().join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

/// The accepted difference between the requested bindgen ABI and the installed
/// libpdfium build, as `(bindgen, binary)`.
///
/// `None` means the two are expected to name the same PDFium. Anything else is
/// a gap somebody has looked at and written down, and the pair is exact on
/// purpose: move either pin and this stops matching, which is the whole point.
/// Whoever moves one has to come back here, redo the symbol-and-header
/// comparison in the module docs above, and either record the new pair or
/// delete this because the two finally converged.
const DECLARED_GAP: Option<(&str, &str)> = None;

/// The `pdfium-render = { ... }` declaration from `Cargo.toml`, flattened.
///
/// Scoped to the declaration rather than the whole file because the
/// surrounding comments walk through the ABI's history and name `pdfium_7543`,
/// `pdfium_7763` and `pdfium_latest` in prose. A scan over the file would read
/// those as pins.
fn pdfium_render_declaration() -> String {
    let manifest = read("Cargo.toml");
    let start = manifest
        .find("\npdfium-render = {")
        .map(|i| i + 1)
        .expect("a `pdfium-render = {` declaration in Cargo.toml");
    let rest = &manifest[start..];
    let end = rest
        .find("}\n")
        .map(|i| i + 1)
        .expect("the pdfium-render declaration is closed");
    rest[..end].split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The milestone in the first `pdfium-<digits>` of `haystack`, if it has one.
///
/// The separator is what tells the two kinds of pin apart and it is worth
/// stating: a bindgen feature is `pdfium_7881` with an underscore and a
/// release tag is `pdfium-8054` with a hyphen. Neither pattern can match the
/// other, so a file naming both (`README.md` does) reads correctly to both.
///
/// This one returns an `Option` because its caller scans `README.md` line by
/// line and most lines have no pin on them. Everything else goes through
/// [`sole_milestone`], where an absence is a failure rather than a skip.
fn build_milestone(haystack: &str) -> Option<String> {
    all_milestones_after(haystack, "pdfium-").into_iter().next()
}

/// Every `<prefix><digits>` milestone in `haystack`, in order, with duplicates
/// kept.
///
/// Enumerating rather than taking the first match is not tidiness, it is the
/// bug this function was written twice for. The contract test's assertions read
///
/// ```text
/// !decl.contains("pdfium_latest"),
///  let named = decl.contains("pdfium_7881");
/// ```
///
/// and a "first line that mentions the literal" search lands on `pdfium_latest`,
/// which carries no digits. The extractor then returned nothing and the guard
/// failed claiming the contract test names no milestone, which is the opposite
/// of true. A scan that finds one thing and reads nothing out of it has to keep
/// looking rather than conclude.
fn all_milestones_after(haystack: &str, prefix: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut from = 0usize;
    while let Some(i) = haystack[from..].find(prefix) {
        let at = from + i + prefix.len();
        let digits: String = haystack[at..]
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        if !digits.is_empty() {
            out.push(digits);
        }
        from = at.max(from + i + 1);
    }
    out
}

/// The one milestone `haystack` names, or a panic saying which way it failed.
///
/// Zero and "more than one distinct" are different mistakes and neither may
/// pass: an extractor that reads nothing would make the agreement checks
/// vacuously true, and one that reads two would have to pick, which is the
/// guess this whole file exists to remove.
fn sole_milestone(haystack: &str, prefix: &str, what: &str) -> String {
    let found = all_milestones_after(haystack, prefix);
    assert!(
        !found.is_empty(),
        "{what} names no `{prefix}<digits>` milestone, so nothing was read out \
         of it and every comparison against it would be vacuous"
    );
    let mut distinct: Vec<&String> = found.iter().collect();
    distinct.sort();
    distinct.dedup();
    assert!(
        distinct.len() == 1,
        "{what} names more than one `{prefix}<digits>` milestone ({distinct:?}), \
         so there is no single pin here to compare"
    );
    found[0].clone()
}

/// Every place naming the libpdfium *build*, as `(what it is, milestone)`.
///
/// Each entry fails loudly when its anchor is not found. An extractor that
/// silently returns nothing would make the agreement below vacuously true,
/// which is the failure this whole file is about.
fn declared_builds() -> Vec<(&'static str, String)> {
    let dockerfile = read("tools/Dockerfile.ci");
    let arg_line = dockerfile
        .lines()
        .find(|l| l.trim_start().starts_with("ARG PDFIUM_RELEASE="))
        .expect("tools/Dockerfile.ci declares `ARG PDFIUM_RELEASE=`");
    let image = sole_milestone(
        arg_line,
        "pdfium-",
        "tools/Dockerfile.ci's ARG PDFIUM_RELEASE",
    );

    let readme = read("README.md");
    let mut out = vec![("tools/Dockerfile.ci ARG PDFIUM_RELEASE", image)];
    let mut seen = 0usize;
    for line in readme.lines() {
        if let Some(m) = build_milestone(line) {
            seen += 1;
            out.push(("README.md", m));
        }
    }
    assert!(
        seen >= 2,
        "README.md names {seen} `pdfium-<digits>` builds and it documents two \
         download URLs plus the prose beside them, so this scan is not \
         reading the install instructions it means to"
    );
    out
}

/// Every place naming the requested bindgen ABI, as `(what it is, milestone)`.
///
/// Three files repeat this literal and nothing relates them. `publish.yml`'s
/// copy is the one that matters most and reads least like a pin: it is a shell
/// `case` pattern whose failure message says "no pdfium_XXXX feature is
/// named", which is true of an unpinned manifest and equally true of a
/// manifest pinned to a *newer* ABI. Bump `Cargo.toml` alone and the release
/// gate refuses the upload while reporting a reason that is not the reason.
fn declared_abis() -> Vec<(&'static str, String)> {
    let manifest = sole_milestone(
        &pdfium_render_declaration(),
        "pdfium_",
        "Cargo.toml's pdfium-render feature list",
    );

    let publish = read(".github/workflows/publish.yml");
    let publish_cases: String = publish
        .lines()
        .filter(|l| l.contains("*pdfium_") && l.contains(')'))
        .collect::<Vec<_>>()
        .join("\n");
    let publish_abi = sole_milestone(
        &publish_cases,
        "pdfium_",
        ".github/workflows/publish.yml's `*pdfium_...*)` case patterns",
    );

    let contract = read("tests/pdfium_dependency_contract.rs");
    let contract_assertions: String = contract
        .lines()
        .filter(|l| l.contains("decl.contains(\"pdfium_"))
        .collect::<Vec<_>>()
        .join("\n");
    let contract_abi = sole_milestone(
        &contract_assertions,
        "pdfium_",
        "tests/pdfium_dependency_contract.rs's `decl.contains(\"pdfium_...\")` assertions",
    );

    vec![
        ("Cargo.toml pdfium-render features", manifest),
        (".github/workflows/publish.yml case pattern", publish_abi),
        ("tests/pdfium_dependency_contract.rs assertion", contract_abi),
    ]
}

/// Every file that installs a libpdfium installs the same one.
///
/// `tools/Dockerfile.ci` builds the image `make ci` runs in and `README.md`
/// tells a human what to install by hand. They were one release apart until
/// #1012, when the image sat on 7881 while everything else had moved, and the
/// comment that was supposed to keep them in step pointed at a file that could
/// not have said so.
#[test]
#[cfg_attr(miri, ignore)] // reads files from the tree, and Miri isolates the filesystem
fn every_place_that_installs_a_libpdfium_installs_the_same_build() {
    let declared = declared_builds();
    let first = declared[0].1.clone();
    let disagreeing: Vec<&(&str, String)> = declared.iter().filter(|(_, m)| *m != first).collect();
    assert!(
        disagreeing.is_empty(),
        "these name different libpdfium builds, so the image `make ci` uses \
         and the one the README tells a human to install are not the same \
         PDFium: expected {first} everywhere, found {disagreeing:?} among \
         {declared:?}"
    );
}

/// Every file that names the requested bindgen ABI names the same one.
///
/// This is the copy that bites at release time rather than in CI, so it is the
/// one least likely to be caught by hand.
#[test]
#[cfg_attr(miri, ignore)] // reads files from the tree, and Miri isolates the filesystem
fn every_place_that_names_the_bindgen_abi_names_the_same_one() {
    let declared = declared_abis();
    let manifest = declared[0].1.clone();
    let disagreeing: Vec<&(&str, String)> = declared
        .iter()
        .filter(|(_, m)| *m != manifest)
        .collect();
    assert!(
        disagreeing.is_empty(),
        "Cargo.toml requests pdfium_{manifest} and these do not agree, so the \
         release gate and the contract test are asserting against an ABI the \
         manifest no longer asks for: {disagreeing:?} among {declared:?}"
    );
}

/// The bindgen ABI and the installed build are the same PDFium, or the gap is
/// the one somebody wrote down.
///
/// This is the assertion #1017 asked for. Before it there was no file holding
/// both numbers, so "7881 bindings against an 8054 binary" was not a state
/// anything could be in disagreement about.
#[test]
#[cfg_attr(miri, ignore)] // reads files from the tree, and Miri isolates the filesystem
fn the_bindgen_abi_and_the_installed_build_agree_or_the_gap_is_declared() {
    let abi = declared_abis()[0].1.clone();
    let build = declared_builds()[0].1.clone();

    match DECLARED_GAP {
        None => assert_eq!(
            abi, build,
            "Cargo.toml requests pdfium-render's `pdfium_{abi}` bindings and \
             tools/Dockerfile.ci installs libpdfium `pdfium-{build}`, so the \
             bindgen set and the library are different PDFium milestones and \
             nothing in the tree says so. pdfium-render is bound dynamically \
             here, so this is not a link error: it is a dlsym against a \
             library the bindings were not generated from. Either pin both to \
             one milestone, or measure the gap (symbol exports and the public \
             headers of both builds) and record the pair in DECLARED_GAP with \
             what makes it safe."
        ),
        Some((want_abi, want_build)) => {
            assert_ne!(
                want_abi, want_build,
                "DECLARED_GAP records {want_abi} against {want_build}, which \
                 is not a gap. Use None when the two agree."
            );
            assert_eq!(
                (abi.as_str(), build.as_str()),
                (want_abi, want_build),
                "the declared gap is pdfium_{want_abi} bindings against \
                 pdfium-{want_build}, but the tree now has pdfium_{abi} \
                 against pdfium-{build}. One of the two pins moved, so the \
                 measurement in this file's module docs no longer describes \
                 what is being built. Redo it and update DECLARED_GAP, or \
                 drop DECLARED_GAP if they have converged."
            );
        }
    }
}
