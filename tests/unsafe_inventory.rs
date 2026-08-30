//! Holds `merge-gate.yml`'s statement about this crate's own `unsafe` to the tree.
//!
//! That comment used to say the crate had no `unsafe` at all, "all four matches for
//! the word under `src/` are in comments". There were fifteen matches and ten were
//! real. Nothing checked the claim, so it drifted, and because it is the sentence
//! that explains what Miri is covering here it sent issue #675 looking at the
//! dependencies when the crate has a dav1d FFI boundary of its own (issue #897).
//!
//! This pins the **set of files** rather than a count, deliberately. A bare number
//! tells a reader that something moved; a set tells them which file and lets them
//! decide whether it should be there. The same reasoning is behind the `#607`
//! countdown in `sample_kind_spine.rs`.
//!
//! # The scan had to read code before it could pin anything
//!
//! Until #943 it did not. The scan tracked `"` strings and not character
//! literals, so the `'"'` in `.split_once('"')` at `src/raster.rs:2540` inverted
//! its string state and everything after it was swallowed as string contents. A
//! real `unsafe { std::ptr::read(&x) }` planted in `src/raster.rs` left this
//! **green**, measured. Blind spans, probing every 25 lines: `src/raster.rs` 2550
//! to 3475, `src/jp2k.rs` 3675 to 6325 (about 2650 lines), `src/imageio.rs` 2800
//! to 3950 and 5050 to 5350, `src/connection.rs` 550 to 600.
//!
//! The answer it gave was right anyway, which is the part worth remembering: a
//! correct scan finds the same one file. **The set being right is not evidence the
//! scan works**, and the control that was supposed to be that evidence tested four
//! comment forms and a string literal and never a character literal, so it passed
//! over the exact hole.
//!
//! The mask is now `tests/common/mask.rs`'s shared
//! [`mask::mask_literals_and_comments`], the same one
//! `tests/sample_kind_spine.rs` calls, so there is no second copy left to fall
//! out of step (issue #968).

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

#[path = "common/mask.rs"]
mod mask;

/// The manifest, at compile time, the way `tests/ci_feature_coverage.rs` pulls
/// it in: the file this asserts about and the binary asserting cannot then
/// drift apart, and editing it forces a rebuild.
const CARGO_TOML: &str = include_str!("../Cargo.toml");

/// Files under `src/` allowed to contain real `unsafe`, and why.
///
/// `avif.rs` is the dav1d FFI boundary: an `unsafe extern "C"` block, a raw pointer
/// `as_ref()`, and pointer arithmetic with an unaligned read. It is behind
/// `#[cfg(feature = "avif")]`, and `default = []`, so a default build (and so a bare
/// `cargo miri test`) compiles all of it out. That last clause is what
/// [`the_crates_own_unsafe_stays_out_of_a_default_build`] holds to the manifest.
const ALLOWED: [(&str, &str); 1] = [(
    "avif.rs",
    "dav1d FFI, behind the non-default `avif` feature",
)];

fn files_with_real_unsafe() -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let files = mask::rs_files_under(&src);
    assert!(
        files.len() > 30,
        "positive control failed: only {} files found under src/, so an empty \
         answer below would be an empty answer about nothing",
        files.len()
    );
    for (rel, path) in files {
        let text = fs::read_to_string(&path).expect("read source file");
        let code = mask::mask_literals_and_comments(&text);
        if code
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .any(|w| w == "unsafe")
        {
            found.insert(rel);
        }
    }
    found
}

/// The `[features]` table as a map from a feature to the features it enables,
/// with `dep:` and `crate/feature` entries dropped.
///
/// Structurally parsed with the `toml` crate (issue #968) rather than a
/// hand-rolled line-based state machine: the state machine this replaced
/// tracked `[section]` headers and multi-line array continuation by hand,
/// which is a strict superset in complexity of the plain line-grep #949
/// proved insufficient, on code that gates whether `unsafe` FFI enters a
/// default build. `[dep:x` and `crate/feature` filtering stays bespoke in
/// [`enabled_features`], since that decision is about this crate's own
/// features and not something a general TOML parser knows.
fn feature_table(manifest: &str) -> std::collections::BTreeMap<String, Vec<String>> {
    let doc: toml::Value = manifest.parse().expect("Cargo.toml must be valid TOML");
    let features = doc
        .get("features")
        .and_then(toml::Value::as_table)
        .cloned()
        .unwrap_or_default();
    features
        .into_iter()
        .map(|(name, value)| {
            let entries = value
                .as_array()
                .unwrap_or_else(|| panic!("[features].{name} is not an array in Cargo.toml"));
            (name, enabled_features(entries))
        })
        .collect()
}

/// The feature names inside one `[..]` list, dropping `dep:x` and
/// `crate/feature` entries, which turn on a dependency rather than a feature
/// of this crate.
fn enabled_features(list: &[toml::Value]) -> Vec<String> {
    list.iter()
        .map(|v| {
            v.as_str()
                .unwrap_or_else(|| panic!("a [features] list entry is not a string: {v:?}"))
        })
        .filter(|s| !s.starts_with("dep:") && !s.contains('/'))
        .map(str::to_owned)
        .collect()
}

/// The features a build enables, resolved **transitively** from `roots`.
///
/// Transitive because a feature list is a graph and the guard below is about
/// what a default build compiles. `default = ["pdfium"]` with
/// `pdfium = ["dep:pdfium-render", "avif"]` puts `avif` in a default build
/// without the string "avif" appearing anywhere near `default`, and the line
/// grep this replaces was green over exactly that (issue #949).
fn feature_closure(manifest: &str, roots: &[&str]) -> BTreeSet<String> {
    let table = feature_table(manifest);
    // Seeded with the roots themselves, not left empty: a cycle that loops
    // back to touch a root by name would otherwise reprocess it once before
    // the first successful `seen.insert` for it closes the loop. Harmless
    // for this crate's actual, acyclic `Cargo.toml`, but a graph-walk
    // function claiming to resolve "a feature list is a graph" should not
    // carry an asymmetry a hand-edited manifest could expose (issue #940's
    // panel).
    let mut seen: BTreeSet<String> = roots.iter().map(|r| (*r).to_owned()).collect();
    let mut queue: Vec<String> = roots.iter().map(|r| (*r).to_owned()).collect();
    while let Some(name) = queue.pop() {
        for next in table.get(&name).map(Vec::as_slice).unwrap_or_default() {
            if seen.insert(next.clone()) {
                queue.push(next.clone());
            }
        }
    }
    seen
}

/**
 * Tests that the default-build feature set is resolved through `[features]`
 * rather than read off one line.
 * A feature list is a graph: `default = ["pdfium"]` with
 * `pdfium = ["dep:pdfium-render", "avif"]` puts `avif` in a default build
 * without the string "avif" appearing anywhere near `default`, and
 * `the_crates_own_unsafe_stays_out_of_a_default_build` greps that one line
 * (issue #949). `ci_feature_coverage.rs` would not catch it either: its
 * `declared_features` drops `default` and never reads a feature's dependency
 * array.
 * Works by resolving a planted manifest that has exactly that shape, then
 * resolving the real one from a root whose answer is known and not empty, so
 * a resolver that returns nothing fails here rather than passing.
 * Input: a planted manifest and this crate's -> Output: `avif` reachable in
 * the first, `pdfium` reachable from `pdfium-static` in the second.
 */
#[test]
fn the_default_build_feature_set_is_resolved_through_the_graph() {
    const PLANTED: &str = "[package]\nname = \"x\"\n\n[features]\n\
                           default = [\"pdfium\"]\n\
                           pdfium = [\"dep:pdfium-render\", \"avif\"]\n\
                           avif = [\"dep:rav1d\"]\n";
    let planted = feature_closure(PLANTED, &["default"]);
    assert!(
        planted.contains("avif"),
        "`default = [\"pdfium\"]` with `pdfium = [.., \"avif\"]` puts `avif` \
         in a default build, and the resolver did not see it: {planted:?}"
    );

    // The control that cannot pass on nothing: a root in the real manifest
    // whose closure is known and non-empty.
    let real = feature_closure(CARGO_TOML, &["pdfium-static"]);
    assert!(
        real.contains("pdfium"),
        "`pdfium-static = [\"pdfium\", ..]`, so the closure must carry \
         `pdfium`; got {real:?}"
    );

    // A cycle that loops back to touch the root by name. Nothing in this
    // crate's real `Cargo.toml` has one, so nothing else exercises the
    // graph-walk on one; a hand-edited manifest is not ruled out by the
    // function's own contract, and the walk must still terminate with the
    // right closure rather than hang or lose a feature (issue #940's panel).
    const CYCLIC: &str = "[package]\nname = \"x\"\n\n[features]\n\
                          a = [\"b\"]\npdfium = []\nb = [\"a\", \"pdfium\"]\n";
    let cyclic = feature_closure(CYCLIC, &["a"]);
    assert_eq!(
        cyclic,
        ["a", "b", "pdfium"]
            .into_iter()
            .map(str::to_owned)
            .collect(),
        "a cycle back to the root must still terminate with the full closure; got {cyclic:?}"
    );
}

#[cfg_attr(miri, ignore)]
#[test]
fn only_the_named_files_carry_the_crates_own_unsafe() {
    let found = files_with_real_unsafe();
    let allowed: BTreeSet<String> = ALLOWED.iter().map(|(f, _)| (*f).to_owned()).collect();
    assert_eq!(
        found, allowed,
        "the set of files under src/ carrying real `unsafe` has moved.\n\
         If a file gained `unsafe`, add it to ALLOWED with the reason, and check whether \
         merge-gate.yml's paragraph about Miri coverage is still true (issue #897).\n\
         If a file lost it, drop the row."
    );
}

#[test]
fn the_scanner_reads_code_and_not_prose() {
    // The positive control: a real block is found.
    assert!(mask::mask_literals_and_comments("fn f() { unsafe { g() } }").contains("unsafe"));
    let mut lost: Vec<String> = Vec::new();
    // And a real block is still found after each literal form that can flip
    // the scanner's state and swallow the rest of the file. The first row is
    // the one that was live: `src/raster.rs:2540` spells `.split_once('"')`,
    // and a scan that does not know character literals reads that `"` as
    // opening a string, so everything from line 2550 to the end of the file
    // was string contents. Measured blind spans at the time: `src/raster.rs`
    // 2550 to 3475, `src/jp2k.rs` 3675 to 6325, about 2650 lines, plus
    // stretches of `src/imageio.rs` and `src/connection.rs` (issue #943).
    for (label, code) in [
        (
            "a char literal holding a quote",
            "fn f(s: &str) { let _ = s.split_once('\"'); }\nfn g() { unsafe { h() } }\n",
        ),
        (
            "a char literal whose escape is a quote, beside one that holds a quote",
            "fn f(s: &str) { let _ = (s.split_once('\\''), s.split_once('\"')); }\n\
             fn g() { unsafe { h() } }\n",
        ),
        (
            "a raw string holding an unmatched quote",
            "fn f() { let _ = r#\"an unmatched \" quote\"#; }\nfn g() { unsafe { h() } }\n",
        ),
        (
            "a lifetime on the line the block is on",
            "fn f<'a>(s: &'a str) -> bool { let t: &'static str = \"n\"; unsafe { h(t) } }\n",
        ),
        (
            "a raw C-string holding an unmatched quote, issue #940's panel",
            "fn f() { let _ = cr#\"an unmatched \" quote\"#; }\nfn g() { unsafe { h() } }\n",
        ),
    ] {
        // Collected rather than asserted row by row, so one run names every
        // literal form the scanner loses the file to instead of the first.
        if !mask::mask_literals_and_comments(code).contains("unsafe") {
            lost.push(format!("{label}:\n{code}"));
        }
    }
    assert!(
        lost.is_empty(),
        "the scanner lost the rest of the file to {} literal form(s):\n{}",
        lost.len(),
        lost.join("\n")
    );
    // The negative controls, one per comment form, because pixel.rs:423 is a comment
    // that argues *for* a rule and a flat grep would have flagged it as a breach.
    for prose in [
        "// an unsafe tile path\n",
        "//! forbid(unsafe_code) at these settings\n",
        "/* hand-rolled unsafe */",
        "/** rather than hand-rolled `unsafe` */",
        "let s = \"unsafe\";",
    ] {
        assert!(
            !mask::mask_literals_and_comments(prose).contains("unsafe"),
            "the scanner read prose as code: {prose}"
        );
    }
}

#[test]
fn the_crates_own_unsafe_stays_out_of_a_default_build() {
    // The claim that survives from merge-gate.yml's old paragraph: a default build,
    // and so a bare `cargo miri test`, reaches none of this crate's own `unsafe`.
    let default = feature_closure(CARGO_TOML, &["default"]);
    for (file, _) in ALLOWED {
        let feature = file.trim_end_matches(".rs");
        assert!(
            !default.contains(feature),
            "`{feature}` is reachable from the default feature list, so a default build \
             now compiles this crate's own `unsafe`. merge-gate.yml's paragraph on Miri \
             coverage is no longer true and must be rewritten (issue #897).\n  \
             default closure = {default:?}"
        );
    }
}

/**
 * Tests that the walk this inventory is built on descends into
 * subdirectories.
 * `src/` is flat today, so the file-count control cannot notice a walk that
 * stops at the top: an `unsafe` block in the first `src/anything/mod.rs`
 * anybody adds would be invisible and this guard would report a clean tree
 * (issue #949). `fuzz/` is the directory in this repo that already has a
 * nested `.rs`, so the control uses that rather than a fixture nobody would
 * maintain.
 * Works by walking `fuzz/` and asserting the nested target turns up under its
 * subdirectory path.
 * Input: `fuzz/` -> Output: `fuzz_targets/fuzz_fits.rs` is in the set.
 */
#[cfg_attr(miri, ignore)]
#[test]
fn the_walk_descends_into_subdirectories() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("fuzz");
    let found: Vec<String> = mask::rs_files_under(&root)
        .into_iter()
        .map(|(r, _)| r)
        .collect();
    assert!(
        found.contains(&"fuzz_targets/fuzz_fits.rs".to_owned()),
        "the walk did not descend into `fuzz/fuzz_targets/`, so a module in a \
         subdirectory of `src/` would be invisible to this inventory and it \
         would report a clean tree. Found: {found:?}"
    );
}
