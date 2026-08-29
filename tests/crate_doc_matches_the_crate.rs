//! The four lists in `README.md` and the crate root have to match the lists the
//! code already keeps (issue #950).
//!
//! `src/lib.rs` is the docs.rs front page and it named **five** of the twelve
//! features this crate declares, missing four that gate headline codec
//! capability. `README.md` said the crate decodes "JPEG, PNG, TIFF via the
//! `image` crate" when the sniffer has seventeen containers in it, listed 21 of
//! 61 public modules, described the pre-#844 gate, and gave the `pixel` module
//! four carriers when `PixelFormat` has fourteen and the new ones are this
//! release's headline break.
//!
//! Every one of those is prose enumerating something the code enumerates too,
//! with nothing connecting the two. That is the shape #881 fixed for
//! `encode_to_target`'s format list, and this is the same fix applied to the
//! four lists a new user reads first.
//!
//! # There is no vacuous state
//!
//! Each check is a **set equality** against a list the compiler or cargo
//! already maintains, and each parser carries a positive control. If a parser
//! stops finding anything, its side of the equality is empty and the other side
//! is not, so the check fails rather than passing on nothing. That is the
//! subset-over-an-empty-set trap `tests/changelog_preamble.rs` was caught by,
//! avoided by never writing a subset assertion in the first place.
//!
//! Everything here is `include_str!` at compile time rather than read at
//! runtime, so these stay runnable under Miri and off
//! `tests/miri_fs_test_inventory.txt`. The one exception walks `src/` and says
//! so.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

const CARGO_TOML: &str = include_str!("../Cargo.toml");
const LIB_RS: &str = include_str!("../src/lib.rs");
const README: &str = include_str!("../README.md");
const SOURCE_RS: &str = include_str!("../src/source.rs");
const PIXEL_RS: &str = include_str!("../src/pixel.rs");
const MAKEFILE: &str = include_str!("../Makefile");
const MIGRATION: &str = include_str!("../MIGRATION.md");

// ---------------------------------------------------------------------------
// Parsers over the lists the code keeps
// ---------------------------------------------------------------------------

/// The feature names in `Cargo.toml`'s `[features]`, minus `default`.
///
/// Same parse as `tests/ci_feature_coverage.rs`, which holds the same table
/// against CI and the Makefile. That one answers "does every feature get
/// linted and tested"; this one answers "does every feature get documented".
fn declared_features() -> BTreeSet<&'static str> {
    let mut out = BTreeSet::new();
    let mut inside = false;
    for line in CARGO_TOML.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            inside = trimmed == "[features]";
            continue;
        }
        if !inside || trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some((name, _)) = trimmed.split_once('=') else {
            continue;
        };
        let name = name.trim();
        if name != "default" && !name.is_empty() {
            out.insert(name);
        }
    }
    out
}

/// Every `pub mod` in the crate root, feature-gated or not.
fn public_modules() -> BTreeSet<&'static str> {
    LIB_RS
        .lines()
        .filter_map(|l| l.trim().strip_prefix("pub mod "))
        .filter_map(|rest| rest.strip_suffix(';'))
        .collect()
}

/// The containers `sniff` recognises, read off `SniffedFormat::next`.
///
/// That chain is an exhaustive `match`, so the compiler refuses to build a
/// tree where it has fallen behind the enum. Reading it here rather than the
/// enum body means this guard inherits that enforcement instead of adding a
/// third hand-written list.
fn sniffed_containers() -> BTreeSet<&'static str> {
    let start = SOURCE_RS
        .find("const fn next(self) -> Option<Self> {")
        .expect("`SniffedFormat::next` lives in src/source.rs");
    let body = &SOURCE_RS[start..];
    let end = body
        .find("\n    }\n")
        .expect("`next` closes at four-space indentation");
    let mut out = BTreeSet::new();
    for line in body[..end].lines() {
        let trimmed = line.trim();
        let Some(rest) = trimmed.strip_prefix("Self::") else {
            continue;
        };
        let name: &str = rest.split([' ', '=']).next().unwrap_or("");
        if !name.is_empty() {
            out.insert(name);
        }
        if let Some(i) = rest.find("Some(Self::") {
            let tail = &rest[i + "Some(Self::".len()..];
            if let Some(j) = tail.find(')') {
                out.insert(&tail[..j]);
            }
        }
    }
    out
}

/// The `PixelFormat` carriers.
fn pixel_carriers() -> BTreeSet<&'static str> {
    let start = PIXEL_RS
        .find("pub enum PixelFormat {")
        .expect("`PixelFormat` lives in src/pixel.rs");
    let body = &PIXEL_RS[start..];
    let end = body.find("\n}\n").expect("the enum closes at column zero");
    let mut out = BTreeSet::new();
    for line in body[..end].lines() {
        // Variants sit at exactly four spaces; doc comments and attributes do
        // not start with an uppercase letter.
        let Some(rest) = line.strip_prefix("    ") else {
            continue;
        };
        if !rest.starts_with(|c: char| c.is_ascii_uppercase()) {
            continue;
        }
        let name: &str = rest.split(['(', ',', ' ', '{']).next().unwrap_or("");
        if !name.is_empty() {
            out.insert(name);
        }
    }
    out
}

/// The features `make clippy` lints, from `LINTED_FEATURES` in the Makefile.
fn linted_features() -> BTreeSet<&'static str> {
    let line = MAKEFILE
        .lines()
        .find(|l| l.starts_with("LINTED_FEATURES"))
        .expect("the Makefile declares LINTED_FEATURES");
    line.split_once(":=")
        .expect("LINTED_FEATURES is a := assignment")
        .1
        .split_whitespace()
        .collect()
}

// ---------------------------------------------------------------------------
// Parsers over the prose
// ---------------------------------------------------------------------------

/// The section of a markdown document (or of a `//!` crate doc) that starts at
/// `heading` and stops at the next heading of the same level or shallower.
///
/// `heading` carries its own `#` marks and, for a crate doc, its `//! ` prefix,
/// so the nesting level comes from the argument rather than a second parameter
/// somebody can get wrong.
fn section<'a>(doc: &'a str, heading: &str) -> &'a str {
    let at = doc
        .find(heading)
        .unwrap_or_else(|| panic!("{heading:?} is a heading in this document"));
    let doc_comment = heading.starts_with("//! ");
    let hashes = heading
        .trim_start_matches("//! ")
        .chars()
        .take_while(|c| *c == '#')
        .count();
    let body = &doc[at + heading.len()..];
    let mut end = body.len();
    for level in 1..=hashes {
        let marker = if doc_comment {
            format!("\n//! {} ", "#".repeat(level))
        } else {
            format!("\n{} ", "#".repeat(level))
        };
        if let Some(i) = body.find(&marker) {
            end = end.min(i);
        }
    }
    &body[..end]
}

/// Every ``**`name`**`` in `text`: a bolded code span, which is how the crate
/// root writes a feature name.
fn bold_code_spans(text: &str) -> BTreeSet<&str> {
    let mut out = BTreeSet::new();
    let mut rest = text;
    while let Some(i) = rest.find("**`") {
        rest = &rest[i + 3..];
        let Some(j) = rest.find("`**") else { break };
        out.insert(&rest[..j]);
        rest = &rest[j + 3..];
    }
    out
}

/// The first cell of every table row in `text`, when that cell is a code span.
fn first_column_code_spans(text: &str) -> BTreeSet<&str> {
    let mut out = BTreeSet::new();
    for line in text.lines() {
        let Some(rest) = line.trim().strip_prefix("| `") else {
            continue;
        };
        if let Some(j) = rest.find('`') {
            out.insert(&rest[..j]);
        }
    }
    out
}

/// Split a prose list ("a, b, c and d") into its items, dropping any
/// parenthetical.
fn prose_list(text: &str) -> BTreeSet<&str> {
    let mut out = BTreeSet::new();
    for chunk in text.split(", ") {
        for item in chunk.split(" and ") {
            let item = item.split(" (").next().unwrap_or(item).trim();
            if !item.is_empty() {
                out.insert(item);
            }
        }
    }
    out
}

/// The prose spelling each sniffed container has to appear under.
///
/// A third list, and it cannot drift: the first assertion in
/// `the_readme_decode_list_names_every_container_the_sniffer_has` is that its
/// keys equal [`sniffed_containers`], so a new variant fails here before it
/// can fail against the README.
const CONTAINER_SPELLINGS: &[(&str, &str)] = &[
    ("Analyze", "Analyze"),
    ("Avif", "AVIF"),
    ("Fits", "FITS"),
    ("Gif", "GIF"),
    ("Jp2k", "JPEG 2000"),
    ("Jpeg", "JPEG"),
    ("Jxl", "JPEG XL"),
    ("Mat", "MATLAB"),
    ("Netpbm", "Netpbm"),
    ("Nifti", "NIfTI"),
    ("OpenExr", "OpenEXR"),
    ("Png", "PNG"),
    ("Radiance", "Radiance"),
    ("Tiff", "TIFF"),
    ("Uhdr", "Ultra HDR"),
    ("Vips", "`.v`"),
    ("WebP", "WebP"),
];

/// Every `src/**/*.rs` file.
///
/// Recursive, because `read_dir` alone stops at the top level and a module that
/// moved into a subdirectory would leave the two walking checks below reading
/// nothing about it.
fn rust_sources() -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("{} must be readable: {e}", dir.display()))
            .map(|e| e.expect("a readable directory entry").path())
            .collect();
        paths.sort();
        for path in paths {
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(path);
            }
        }
    }
    let mut out = Vec::new();
    walk(&Path::new(env!("CARGO_MANIFEST_DIR")).join("src"), &mut out);
    assert!(
        out.len() > 50,
        "the walk found only {} source files, so it is looking in the wrong place",
        out.len()
    );
    out
}

/// The part of a file above its `#[cfg(test)]` module, so a scan reads shipping
/// code rather than fixtures.
fn non_test_body(src: &str) -> &str {
    match src.find("\n#[cfg(test)]\n") {
        Some(i) => &src[..i],
        None => src,
    }
}

// ---------------------------------------------------------------------------
// The checks
// ---------------------------------------------------------------------------

/**
 * Tests that the crate root's feature list names every feature `Cargo.toml`
 * declares, and nothing else (issue #950).
 *
 * `src/lib.rs` is the docs.rs front page, so its list is the one a new user
 * reads. It named `pdfium`, `pdfium-static`, `object-store-sink`, `s3`,
 * `tracing` and `packfile`, and missed `avif`, `svg`, `jxl`, `jp2k`, `serde`
 * and `test-util`. Four of the six gate headline codec capability.
 *
 * Equality in both directions: a feature added to `Cargo.toml` without a
 * bullet is red, and a bullet naming something that is not a feature is red
 * too. The bold-code-span spelling is the marker, which is what the section
 * already used.
 */
#[test]
fn the_crate_root_feature_list_names_every_declared_feature() {
    let declared = declared_features();
    assert!(
        declared.len() >= 10,
        "the [features] parser found only {declared:?}"
    );

    let documented = bold_code_spans(section(LIB_RS, "//! ## Feature flags"));
    assert!(
        documented.len() >= 10,
        "the crate-root parser found only {documented:?}, so it has stopped \
         reading the section"
    );
    assert_eq!(
        documented,
        declared.iter().copied().collect::<BTreeSet<_>>(),
        "`src/lib.rs`'s feature list and `Cargo.toml`'s [features] disagree; \
         lib.rs is the docs.rs front page (issue #950)"
    );
}

/**
 * Tests that the README's feature table names every feature too (issue #950).
 *
 * Same list, second copy, and it had the same hole: `avif`, `serde` and
 * `test-util` were missing. Two copies of one list is what #881 argued
 * against, and the argument does not reach here: a README and a docs.rs front
 * page are read by different people in different places, so both have to
 * exist. What they cannot do is disagree, which is what this makes
 * impossible.
 */
#[test]
fn the_readme_feature_table_names_every_declared_feature() {
    let declared: BTreeSet<&str> = declared_features();
    let table = section(README, "## Features\n\n| Feature | Default | Description |");
    let documented = first_column_code_spans(table);
    assert!(
        documented.len() >= 10,
        "the README feature-table parser found only {documented:?}"
    );
    assert_eq!(
        documented, declared,
        "`README.md`'s feature table and `Cargo.toml`'s [features] disagree"
    );
}

/**
 * Tests that the README's module tables name every public module (issue #950).
 *
 * They listed 21 of 61, and none of the roughly thirty format and operation
 * modules, so a reader looking for the WebP lane or the convolution family
 * found no sign that either exists.
 */
#[test]
fn the_readme_module_tables_name_every_public_module() {
    let modules = public_modules();
    assert!(
        modules.len() >= 50,
        "the `pub mod` scan found only {} modules",
        modules.len()
    );

    let tables = section(README, "## Modules");
    let listed = first_column_code_spans(tables);
    assert_eq!(
        listed, modules,
        "`README.md`'s module tables and `pub mod` in `src/lib.rs` disagree"
    );
}

/**
 * Tests that the README's decode list names every container the sniffer has
 * (issue #950).
 *
 * It said "JPEG, PNG, TIFF via the `image` crate". `SniffedFormat` has
 * seventeen variants and most of them are decoders written in this crate, so
 * the sentence was wrong about the count and about the author.
 *
 * The spelling table is a third list, and the first assertion is that its keys
 * equal the sniffer's variants, so it cannot go stale on its own: a new
 * container fails here before anyone gets as far as the README.
 */
#[test]
fn the_readme_decode_list_names_every_container_the_sniffer_has() {
    let containers = sniffed_containers();
    assert!(
        containers.len() >= 15,
        "the `SniffedFormat::next` parser found only {containers:?}"
    );
    let keys: BTreeSet<&str> = CONTAINER_SPELLINGS.iter().map(|(k, _)| *k).collect();
    assert_eq!(
        keys, containers,
        "`CONTAINER_SPELLINGS` and `SniffedFormat` disagree; add the new \
         container's prose spelling, then the README will tell you where"
    );

    let marker = "rather than the file extension: ";
    let at = README
        .find(marker)
        .expect("the decode bullet names its list after a colon");
    let rest = &README[at + marker.len()..];
    let end = rest.find(". ").expect("the list ends in a full stop");
    let named = prose_list(&rest[..end]);

    let want: BTreeSet<&str> = CONTAINER_SPELLINGS.iter().map(|(_, v)| *v).collect();
    assert_eq!(
        named, want,
        "`README.md`'s decode list and `SniffedFormat` disagree"
    );
}

/**
 * Tests that the README's `pixel` paragraph names every `PixelFormat` carrier
 * (issue #950).
 *
 * The modules table gave it "Gray8, RGB8, RGBA8, 16-bit variants", which was
 * four of fourteen and, worse, left out the signed and 32-bit carriers that
 * are this release's headline breaking change. A reader working out whether
 * the break affects them found no list to check against.
 */
#[test]
fn the_readme_pixel_paragraph_names_every_carrier() {
    let carriers = pixel_carriers();
    assert!(
        carriers.len() >= 12,
        "the `PixelFormat` parser found only {carriers:?}"
    );

    let para = section(README, "### Pixel formats");
    let after_colon = para
        .split_once(": ")
        .expect("the paragraph lists the carriers after a colon")
        .1;
    let mut named = BTreeSet::new();
    let mut rest = after_colon;
    while let Some(i) = rest.find('`') {
        rest = &rest[i + 1..];
        let Some(j) = rest.find('`') else { break };
        named.insert(&rest[..j]);
        rest = &rest[j + 1..];
    }
    assert_eq!(
        named, carriers,
        "`README.md`'s pixel-format list and `PixelFormat` disagree"
    );
}

/**
 * Tests that the README describes the gate the Makefile actually runs
 * (issue #950).
 *
 * It said clippy runs over "default + `pdfium` feature", which stopped being
 * true at #844 when `LINTED_FEATURES` grew to nine. That exact staleness, in
 * the handover doc rather than the README, is what cost four duplicate issue
 * filings this epic, and the README copy is the one users read.
 *
 * The word "nine" is checked too, because a count in prose is the half that
 * goes stale silently: every count in this repository has a prose twin and the
 * prose never goes red on its own.
 */
#[test]
fn the_readme_describes_the_clippy_gate_the_makefile_runs() {
    let linted = linted_features();
    assert!(
        linted.len() >= 5,
        "the LINTED_FEATURES parser found only {linted:?}"
    );

    let marker = "Since #844 that is the default build plus ";
    let at = README
        .find(marker)
        .expect("the CI bullet names the linted set");
    let rest = &README[at + marker.len()..];
    let end = rest
        .find(", because")
        .expect("the sentence explains why after the list");
    let mut named = BTreeSet::new();
    let mut scan = &rest[..end];
    while let Some(i) = scan.find('`') {
        scan = &scan[i + 1..];
        let Some(j) = scan.find('`') else { break };
        named.insert(&scan[..j]);
        scan = &scan[j + 1..];
    }
    assert_eq!(
        named, linted,
        "`README.md` and `LINTED_FEATURES` disagree about what clippy lints"
    );

    // The spelled-out count beside it, in the `make clippy` line.
    let words = [
        "", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    ];
    let word = words
        .get(linted.len())
        .unwrap_or_else(|| panic!("no spelling for {} features", linted.len()));
    let line = README
        .lines()
        .find(|l| l.starts_with("make clippy"))
        .expect("the README lists the make targets");
    assert!(
        line.contains(word),
        "the `make clippy` line says {line:?} and there are {} linted features",
        linted.len()
    );
}

/**
 * Tests that no doc example under `src/` carries `ignore` (issue #950).
 *
 * Nine did, so nine snippets the modules recommend were never compiled by
 * anything. One of them could not have compiled under any circumstances: the
 * resume module's own "Intended use" built a `#[non_exhaustive]` struct with a
 * struct literal, which is E0639 outside the defining crate, and it sat there
 * as the first thing a reader of that module sees.
 *
 * The fix is not a scanner that looks for that one mistake. It is handing the
 * examples back to the compiler, which finds every mistake including the ones
 * nobody thought to scan for, and this keeps them there. CI runs
 * `cargo test --features packfile` and `--features object-store-sink`, so the
 * three examples in feature-gated modules are compiled too.
 *
 * Two controls, because "no offenders" is satisfied by a scanner that has
 * stopped reading: the runnable-fence count has to stay above the number of
 * examples the tree has, and the ignore detector is run against a planted
 * example that does carry the marker.
 */
#[test]
#[cfg_attr(miri, ignore)] // reads src/ from disk, blocked by Miri isolation
fn no_doc_example_under_src_carries_ignore() {
    fn fences(body: &str) -> (Vec<String>, usize) {
        let mut ignored = Vec::new();
        let mut runnable = 0usize;
        for line in body.lines() {
            let trimmed = line.trim_start();
            let Some(rest) = trimmed
                .strip_prefix("//! ")
                .or_else(|| trimmed.strip_prefix("/// "))
            else {
                continue;
            };
            let Some(attr) = rest.trim().strip_prefix("```") else {
                continue;
            };
            let attr = attr.trim();
            if attr == "ignore" {
                ignored.push(line.trim().to_owned());
            } else if attr.is_empty() {
                runnable += 1;
            }
        }
        (ignored, runnable)
    }

    // The negative control, run first: the detector fires on a planted marker.
    let (planted, _) = fences("/// ```ignore\n/// let x = 1;\n/// ```\n");
    assert_eq!(planted.len(), 1, "the ignore detector does not detect");

    let files = rust_sources();

    let mut offenders = Vec::new();
    let mut runnable_total = 0usize;
    for path in &files {
        let body = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("{} must be readable: {e}", path.display()));
        let (ignored, runnable) = fences(&body);
        runnable_total += runnable;
        for line in ignored {
            offenders.push(format!("{}: {line}", path.display()));
        }
    }

    // The positive control: opening fences are still being found at all. Every
    // example opens and closes with a bare fence, so this counts both.
    assert!(
        runnable_total >= 36,
        "the fence scan found {runnable_total} bare fences under src/, which is \
         fewer than the tree has; it has stopped reading"
    );
    assert!(
        offenders.is_empty(),
        "these doc examples carry `ignore`, so nothing compiles them: \
         {offenders:?}. Write the setup the example needs and let the compiler \
         hold it (issue #950)"
    );
}

/**
 * Tests that `SourceError::PageOutOfRange`'s doc names every container that
 * can produce it (issue #950).
 *
 * It said ``(`"webp"`, `"jxl"`)``, which was right until #845 folded GIF's own
 * `BadPageNumber` into this variant and gave `resolve_page_range` a third
 * caller. The doc did not move, so the field describing what a caller will see
 * named two of the three things a caller can see.
 *
 * The call sites are the authority, read out of the shipping half of each file
 * so the `"webp"` and `"jxl"` fixtures in `source.rs`'s own test module cannot
 * widen the answer. Set equality, and the call-site scan carries its own
 * control, because a scan that has stopped finding calls agrees perfectly with
 * a doc that has stopped naming containers.
 */
#[test]
#[cfg_attr(miri, ignore)] // reads src/ from disk, blocked by Miri isolation
fn every_multi_page_loader_that_refuses_a_page_is_named_in_the_doc() {
    let mut callers: BTreeSet<String> = BTreeSet::new();
    for path in rust_sources() {
        let body = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{} must be readable: {e}", path.display()));
        let mut rest = non_test_body(&body);
        while let Some(i) = rest.find("resolve_page_range(\"") {
            rest = &rest[i + "resolve_page_range(\"".len()..];
            if let Some(j) = rest.find('"') {
                callers.insert(rest[..j].to_owned());
            }
        }
    }
    assert!(
        callers.len() >= 3,
        "the call-site scan found only {callers:?}, so it has stopped reading"
    );

    let marker = "/// The container, for the message: ";
    let at = SOURCE_RS
        .find(marker)
        .expect("the field doc names the containers after a colon");
    let rest = &SOURCE_RS[at + marker.len()..];
    let end = rest
        .find(", which are")
        .expect("the doc says what the list is after naming it");
    let mut named = BTreeSet::new();
    let mut scan = &rest[..end];
    while let Some(i) = scan.find("`\"") {
        scan = &scan[i + 2..];
        let Some(j) = scan.find("\"`") else { break };
        named.insert(scan[..j].to_owned());
        scan = &scan[j + 2..];
    }
    assert_eq!(
        named, callers,
        "`SourceError::PageOutOfRange`'s doc and `resolve_page_range`'s callers \
         disagree about which containers report it"
    );
}

/**
 * Tests that every MSRV a document states is the one `Cargo.toml` declares
 * (issue #950).
 *
 * `MIGRATION.md` said 1.85 while `rust-version` said 1.97 and the README's
 * badge said 1.97, so a reader picking the wrong file got a floor three minor
 * versions under the real one. Nobody would have found that by reading; it is
 * one number in a closing line of a file about a two-releases-ago migration.
 *
 * The scan is over lines that talk about the floor at all (`MSRV`, `Rust 1.`,
 * or the shields.io rust badge), and every `1.NN` on such a line has to be the
 * manifest's. The control is that at least three such claims are found, which
 * is what the tree has; a scan that stopped matching would otherwise agree
 * with everything.
 */
#[test]
fn every_document_that_states_an_msrv_states_the_manifests() {
    let declared = CARGO_TOML
        .lines()
        .find_map(|l| l.trim().strip_prefix("rust-version = "))
        .map(|v| v.trim().trim_matches('"'))
        .expect("Cargo.toml declares rust-version");
    assert!(
        declared.starts_with("1."),
        "an unexpected rust-version spelling: {declared:?}"
    );

    let mut claims = 0usize;
    for (file, body) in [("README.md", README), ("MIGRATION.md", MIGRATION)] {
        for line in body.lines() {
            if !(line.contains("MSRV")
                || line.contains("Rust 1.")
                || line.contains("badge/rust-1."))
            {
                continue;
            }
            // Every `1.<digits>` on the line.
            let mut rest = line;
            while let Some(i) = rest.find("1.") {
                rest = &rest[i..];
                let len = 2 + rest[2..].chars().take_while(char::is_ascii_digit).count();
                let found = &rest[..len];
                if len > 2 {
                    claims += 1;
                    assert_eq!(
                        found, declared,
                        "{file} states an MSRV of {found}, and `rust-version` is \
                         {declared}: {line}"
                    );
                }
                rest = &rest[len..];
            }
        }
    }
    assert!(
        claims >= 3,
        "the MSRV scan found {claims} claims across the two documents, so it has \
         stopped matching and would agree with any number"
    );
}
