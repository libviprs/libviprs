//! The boundary rules of the DWG provider traits (N4.1, #1072), checked as text.
//!
//! * No ACadSharp, .NET, FFI or `lopdf` type appears in any public signature of
//!   `src/pvec/provider.rs`. The test enumerates the module's public items and
//!   fails on a foreign path, and it proves it can fail by feeding the scanner a
//!   foreign signature first.
//! * The same holds for the public items of the provider module (`mod.rs`) and of
//!   the acadrust provider, whose own `use` of the reader is the one place a
//!   foreign crate is allowed, and only below the signatures.
//! * The acadrust dependency is a git dependency pinned by a full revision, behind
//!   the `dwg-acadrust` feature, and nothing turns it on by default.
//!
//! Nothing here needs a CAD crate, a feature, or a network.

use std::fs;
use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn read(rel: &str) -> String {
    fs::read_to_string(root().join(rel)).unwrap_or_else(|e| panic!("read {rel}: {e}"))
}

/// Drops `//`, `///` and `//!` comments and string literals' contents. Block
/// comments are not used in the files scanned.
fn strip_comments(src: &str) -> String {
    src.lines()
        .map(|l| match l.find("//") {
            Some(i) => &l[..i],
            None => l,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Every public item's header, and every `fn` header (trait methods carry no `pub`
/// of their own): from the keyword to the first `{`, `;` or top-level `,`.
fn public_items(src: &str) -> Vec<String> {
    let text = strip_comments(src)
        .replace("->", " to ")
        .replace("=>", " to ");
    let bytes: Vec<char> = text.chars().collect();
    let starts = |i: usize, kw: &str| -> bool {
        let k: Vec<char> = kw.chars().collect();
        i + k.len() < bytes.len()
            && bytes[i..i + k.len()] == k[..]
            && (i == 0 || !(bytes[i - 1].is_alphanumeric() || bytes[i - 1] == '_'))
            && (bytes[i + k.len()] == ' ' || bytes[i + k.len()] == '(')
    };
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if !(starts(i, "pub") || starts(i, "fn")) {
            i += 1;
            continue;
        }
        let mut depth = 0i32;
        let mut j = i;
        while j < bytes.len() {
            match bytes[j] {
                '<' | '(' | '[' => depth += 1,
                '>' | ')' | ']' => depth -= 1,
                '{' | ';' => break,
                ',' if depth <= 0 => break,
                _ => {}
            }
            j += 1;
        }
        let item: String = bytes[i..j.min(bytes.len())].iter().collect();
        out.push(item.split_whitespace().collect::<Vec<_>>().join(" "));
        i = j.max(i + 1);
    }
    out
}

/// Every `use` statement's text.
fn uses(src: &str) -> Vec<String> {
    let text = strip_comments(src);
    let mut out = Vec::new();
    let mut cur: Option<String> = None;
    for line in text.lines() {
        let t = line.trim();
        if cur.is_none() && (t.starts_with("use ") || t.starts_with("pub use ")) {
            cur = Some(String::new());
        }
        if let Some(c) = cur.as_mut() {
            c.push_str(t);
            c.push(' ');
            if t.ends_with(';') {
                out.push(cur.take().unwrap());
            }
        }
    }
    out
}

/// Path roots that are ours or the standard library's. `thiserror` is in the list
/// because `ProviderError` derives `Error` from it: a derive macro, not a type in any
/// signature, and a dependency the crate already has.
const ALLOWED_ROOTS: [&str; 7] = [
    "crate",
    "std",
    "core",
    "alloc",
    "self",
    "super",
    "thiserror",
];
/// Substrings that are foreign wherever they appear.
const FOREIGN_SUBSTRINGS: [&str; 5] = ["acadsharp", "dotnet", ".net", "*const", "*mut"];
/// Whole words that are foreign (`ffi` is also inside `coefficients`).
const FOREIGN_WORDS: [&str; 5] = ["lopdf", "libc", "c_void", "extern", "ffi"];

/// The names a file brings in from an allowed root (`use std::fmt;` gives `fmt`), so
/// `fmt::Formatter` is not mistaken for a foreign crate's path.
fn local_roots(src: &str) -> Vec<String> {
    uses(src)
        .iter()
        .filter(|u| foreign_use(u).is_none())
        .filter_map(|u| {
            let body = u
                .trim()
                .trim_start_matches("pub ")
                .trim_start_matches("use ")
                .trim();
            let body = body.trim_end_matches(';').trim();
            if body.contains('{') {
                return None;
            }
            body.rsplit("::").next().map(|l| l.trim().to_string())
        })
        .collect()
}

/// What is wrong with one public item header, if anything.
fn foreign_in_item(item: &str, allow_reader_name: bool) -> Option<String> {
    foreign_in_item_with(item, allow_reader_name, &[])
}

fn foreign_in_item_with(item: &str, allow_reader_name: bool, locals: &[String]) -> Option<String> {
    let lower = item.to_lowercase();
    for w in FOREIGN_SUBSTRINGS {
        if lower.contains(w) {
            return Some(format!("`{w}` in `{item}`"));
        }
    }
    for tok in lower.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
        if FOREIGN_WORDS.contains(&tok) {
            return Some(format!("`{tok}` in `{item}`"));
        }
    }
    if !allow_reader_name && lower.contains("acadrust") {
        return Some(format!("`acadrust` in `{item}`"));
    }
    // A path whose first segment is not ours.
    for tok in item.split(|c: char| !(c.is_alphanumeric() || c == '_' || c == ':')) {
        if let Some((first, _)) = tok.split_once("::")
            && !first.is_empty()
            && !ALLOWED_ROOTS.contains(&first)
            && !locals.iter().any(|l| l == first)
        {
            return Some(format!("path `{tok}` in `{item}`"));
        }
    }
    None
}

fn foreign_use(u: &str) -> Option<String> {
    let body = u
        .trim()
        .trim_start_matches("pub ")
        .trim_start_matches("use ")
        .trim();
    let first: String = body
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    if ALLOWED_ROOTS.contains(&first.as_str()) {
        None
    } else {
        Some(format!("`use {body}` roots at `{first}`"))
    }
}

#[test]
fn the_scanner_can_fail() {
    // Positive controls: a scanner that cannot fail proves nothing.
    let bad = [
        "pub fn open(d: acadsharp_rs::Document) -> Result<(), Error>",
        "pub fn decode(&self) -> acs::Item",
        "pub struct Handle { pub raw: *mut u8 }",
        "pub fn load(doc: lopdf::Document)",
        "pub fn run() -> std::ffi::CString",
        "fn next(&mut self) -> *const u8",
        "pub fn get(&self) -> acadrust::CadDocument",
    ];
    for b in bad {
        let items = public_items(b);
        assert!(
            items.iter().any(|i| foreign_in_item(i, false).is_some()),
            "should flag: {b} (items: {items:?})"
        );
    }
    assert!(foreign_use("use acadsharp_rs as acs;").is_some());
    assert!(foreign_use("use crate::cad::{CadSource, ItemHandle};").is_none());
    assert!(foreign_use("use std::fmt;").is_none());
    let good = "pub fn open(source: crate::cad::CadSource<'_>) -> Result<Self, ProviderError>";
    assert!(foreign_in_item(good, false).is_none());
}

#[test]
#[cfg_attr(miri, ignore)] // reads the repository source tree, which Miri isolation blocks
fn the_scanner_enumerates_the_modules_public_items() {
    // If the enumeration silently found nothing, the checks below would pass for the
    // wrong reason. Pin that the whole trait set from the issue is seen.
    let items = public_items(&read("src/pvec/provider.rs"));
    let joined = items.join("\n");
    for name in [
        "trait DwgProvider",
        "trait EntityCursor",
        "struct RawEntity",
        "struct LayerState",
        "struct LayoutState",
        "struct BlockRef",
        "struct SourceTransform",
        "struct ProvenanceStub",
        "struct ProviderCaps",
        "struct InstancePath",
        "fn layouts",
        "fn next_entity",
    ] {
        assert!(
            joined.contains(name),
            "public item `{name}` not enumerated:\n{joined}"
        );
    }
    assert!(items.len() > 60, "only {} public items found", items.len());
}

#[test]
#[cfg_attr(miri, ignore)] // reads the repository source tree, which Miri isolation blocks
fn no_foreign_type_appears_in_any_public_signature_of_the_provider_module() {
    for file in ["src/pvec/provider.rs", "src/pvec/mod.rs"] {
        let src = read(file);
        let mut failures = Vec::new();
        let mut locals = local_roots(&src);
        // `mod.rs` declares the provider module and the feature-gated acadrust and
        // normalized ones by name; naming a child module is not naming a reader's type.
        let is_mod = file.ends_with("mod.rs");
        if is_mod {
            locals.extend([
                "classify".to_string(),
                "fidelity".to_string(),
                "provider".to_string(),
                "dwg_acadrust".to_string(),
                "units".to_string(),
                "pdf_vector".to_string(),
                "dwg_spline".to_string(),
                "normalized".to_string(),
                "view_profile".to_string(),
            ]);
        }
        for item in public_items(&src) {
            if let Some(why) = foreign_in_item_with(&item, is_mod, &locals) {
                failures.push(why);
            }
        }
        for u in uses(&src) {
            let local = locals.iter().any(|l| u.contains(&format!("use {l}::")));
            if let Some(why) = foreign_use(&u)
                && !local
            {
                failures.push(why);
            }
        }
        assert!(
            failures.is_empty(),
            "{file} names a foreign path:\n{}",
            failures.join("\n")
        );
    }
}

#[test]
#[cfg_attr(miri, ignore)] // reads the repository source tree, which Miri isolation blocks
fn the_acadrust_provider_names_no_reader_type_in_a_public_signature() {
    let src = read("src/pvec/dwg_acadrust.rs");
    // Only `pub` items: the private helpers below the signatures are where the reader's
    // own types are allowed, and `fn` headers of trait methods are checked in provider.rs.
    let items: Vec<String> = public_items(&src)
        .into_iter()
        .filter(|i| i.starts_with("pub "))
        .collect();
    assert!(
        items.iter().any(|i| i.contains("struct AcadrustProvider")),
        "the provider is not there:\n{}",
        items.join("\n")
    );
    let mut failures = Vec::new();
    for item in &items {
        if let Some(why) = foreign_in_item(item, true) {
            failures.push(why);
        }
    }
    assert!(
        failures.is_empty(),
        "a reader type is public:\n{}",
        failures.join("\n")
    );
}

#[test]
#[cfg_attr(miri, ignore)] // reads the repository source tree, which Miri isolation blocks
fn the_provider_module_stays_out_of_the_cad_decoder_contract() {
    // CadDecoder and Origin are unchanged by N4.1: the traits reuse two value types
    // and nothing else from `cad`.
    let uses = uses(&read("src/pvec/provider.rs"));
    let joined = uses.join("\n");
    assert!(
        joined.contains("CadSource") && joined.contains("ItemHandle"),
        "{joined}"
    );
    for banned in [
        "CadDecoder",
        "CadDrawing",
        "Origin",
        "Primitive",
        "PrimitiveSink",
    ] {
        assert!(
            !joined.contains(banned),
            "provider.rs imports cad::{banned}"
        );
    }
}

// ---- the dependency pin ----------------------------------------------------------

fn manifest() -> String {
    read("Cargo.toml")
}

fn dependency_line(manifest: &str, key: &str) -> Option<String> {
    manifest
        .lines()
        .find(|l| l.trim_start().starts_with(&format!("{key} =")) && l.contains("git"))
        .map(str::to_string)
}

#[test]
#[cfg_attr(miri, ignore)] // reads the repository source tree, which Miri isolation blocks
fn acadrust_is_a_git_dependency_pinned_by_a_full_revision() {
    let m = manifest();
    let line =
        dependency_line(&m, "acadrust").expect("a `acadrust = { git = ..., rev = ... }` line");
    assert!(
        line.contains("https://github.com/libviprs/acadrust"),
        "{line}"
    );
    let rev = line
        .split("rev = \"")
        .nth(1)
        .and_then(|r| r.split('"').next())
        .unwrap_or_else(|| panic!("no rev in {line}"));
    assert_eq!(
        rev.len(),
        40,
        "a rev is the full 40 hex digits, not a prefix: {rev}"
    );
    assert!(
        rev.chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
        "{rev}"
    );
    for floating in ["branch", "tag"] {
        assert!(
            !line.contains(&format!("{floating} =")),
            "{floating} floats, a rev does not: {line}"
        );
    }
    assert!(
        !line.contains("path ="),
        "a path dependency is not a pin: {line}"
    );
}

#[test]
#[cfg_attr(miri, ignore)] // reads the repository source tree, which Miri isolation blocks
fn acadrust_is_optional_and_only_the_feature_turns_it_on() {
    let m = manifest();
    assert!(
        m.lines()
            .any(|l| l.trim() == "acadrust = { workspace = true, optional = true }"),
        "[dependencies] must take acadrust from the workspace table as optional"
    );
    let feature = m
        .lines()
        .find(|l| l.starts_with("dwg-acadrust ="))
        .expect("a dwg-acadrust feature");
    assert_eq!(feature.trim(), "dwg-acadrust = [\"dep:acadrust\"]");
    let default = m
        .lines()
        .find(|l| l.starts_with("default ="))
        .expect("default features");
    assert_eq!(default.trim(), "default = []");
    for l in m.lines().filter(|l| l.contains("\"dwg-acadrust\"")) {
        assert!(!l.starts_with("default"), "{l}");
    }
}

#[test]
#[cfg_attr(miri, ignore)] // reads the repository source tree, which Miri isolation blocks
fn the_gate_zero_harness_pins_the_same_revision_as_the_crate() {
    let m = manifest();
    let line = dependency_line(&m, "acadrust").expect("acadrust git line");
    let rev = line
        .split("rev = \"")
        .nth(1)
        .and_then(|r| r.split('"').next())
        .expect("rev");
    let harness = read("docs/spikes/acadrust-gate0/results-x64.tsv");
    assert!(
        harness.contains(&format!("# acadrust {rev}")),
        "results-x64.tsv must record the revision it ran against ({rev})"
    );
}
