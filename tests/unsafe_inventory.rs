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
//! Until #943 it did not. [`code_only`] tracked `"` strings and not character
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
//! The mask is now the one `tests/sample_kind_spine.rs` uses, which got character
//! literals right from the start. Keeping the two in step is the cheap half of
//! this; the expensive half was noticing.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

/// The manifest, at compile time, the way `tests/ci_feature_coverage.rs` pulls
/// it in: the file this asserts about and the binary asserting cannot then
/// drift apart, and editing it forces a rebuild.
const CARGO_TOML: &str = include_str!("../Cargo.toml");

/// Files under `src/` allowed to contain real `unsafe`, and why.
///
/// `avif.rs` is the dav1d FFI boundary: an `unsafe extern "C"` block, a raw pointer
/// `as_ref()`, and pointer arithmetic with an unaligned read. It is behind
/// `#[cfg(feature = "avif")]`, and `default = []`, so a default build (and so a bare
/// `cargo miri test`) compiles all of it out.
const ALLOWED: [(&str, &str); 1] = [(
    "avif.rs",
    "dav1d FFI, behind the non-default `avif` feature",
)];

/// Replace comments and the *contents* of every literal with spaces, so the scan
/// reads code and not prose. A scanner that counts the sentence explaining a rule
/// as a breach of that rule is worse than no scanner, and one that reads 2650
/// lines of code as a string is worse than both.
///
/// Deliberately not a Rust parser, and deliberately more than a line-oriented
/// strip. It tracks strings, byte strings, raw strings of any hash depth, and
/// character literals, because each of those can carry a `//` or a `"` that a
/// simpler scan reads as a delimiter and then loses the rest of the file to.
/// That is not hypothetical, it is what this file did until #943: the `'"'` in
/// `.split_once('"')` at `src/raster.rs:2540` inverted the string state, and
/// everything after it was swallowed as string contents while the real code was
/// thrown away. Measured blind spans, probing every 25 lines: `src/raster.rs`
/// 2550 to 3475, `src/jp2k.rs` 3675 to 6325 (about 2650 lines),
/// `src/imageio.rs` 2800 to 3950 and 5050 to 5350, `src/connection.rs` 550 to
/// 600. The inventory's answer was right by luck.
///
/// It is the same masking `tests/sample_kind_spine.rs` runs, deliberately kept
/// in step: that file got character literals right from the start, which is
/// what made this one's blindness a copy away from being avoided.
///
/// A `'` is a character literal only when it really closes one. `&'a str` is a
/// lifetime, and swallowing from there to the next `'` would be the same
/// desynchronisation in a new dress, so [`char_literal_end`] decides.
fn code_only(src: &str) -> String {
    let b = src.as_bytes();
    let n = b.len();
    let mut out: Vec<u8> = Vec::with_capacity(n);
    let mut i = 0usize;
    while i < n {
        if b[i] == b'/' && i + 1 < n && b[i + 1] == b'/' {
            let mut j = i;
            while j < n && b[j] != b'\n' {
                j += 1;
            }
            blank(&mut out, b, i, j);
            i = j;
            continue;
        }
        if b[i] == b'/' && i + 1 < n && b[i + 1] == b'*' {
            let mut j = i + 2;
            let mut depth = 1usize;
            while j < n && depth > 0 {
                if j + 1 < n && b[j] == b'/' && b[j + 1] == b'*' {
                    depth += 1;
                    j += 2;
                } else if j + 1 < n && b[j] == b'*' && b[j + 1] == b'/' {
                    depth -= 1;
                    j += 2;
                } else {
                    j += 1;
                }
            }
            let j = j.min(n);
            blank(&mut out, b, i, j);
            i = j;
            continue;
        }
        if let Some((body, hashes)) = raw_string_start(b, i) {
            // Keep the `r##"` opener and the `"##` closer, blank the body.
            blank_span(&mut out, b, i, body);
            let mut j = body;
            let close = loop {
                if j >= n {
                    break n;
                }
                if b[j] == b'"'
                    && j + hashes < n
                    && b[j + 1..=j + hashes].iter().all(|&c| c == b'#')
                {
                    break j;
                }
                if b[j] == b'"' && hashes == 0 {
                    break j;
                }
                j += 1;
            };
            blank(&mut out, b, body, close.min(n));
            let after = (close + 1 + hashes).min(n);
            blank_span(&mut out, b, close.min(n), after);
            i = after;
            continue;
        }
        if b[i] == b'"' {
            let mut j = i + 1;
            while j < n {
                if b[j] == b'\\' {
                    j += 2;
                    continue;
                }
                if b[j] == b'"' {
                    break;
                }
                j += 1;
            }
            let j = j.min(n);
            out.push(b'"');
            blank(&mut out, b, i + 1, j);
            if j < n {
                out.push(b'"');
            }
            i = j + 1;
            continue;
        }
        if b[i] == b'\''
            && let Some(e) = char_literal_end(b, i)
        {
            out.push(b'\'');
            blank(&mut out, b, i + 1, e - 1);
            out.push(b'\'');
            i = e;
            continue;
        }
        out.push(b[i]);
        i += 1;
    }
    String::from_utf8(out).expect("only ASCII delimiters are rewritten, and each as one byte")
}

/// Push `[from, to)` as spaces, keeping newlines so line numbers survive.
fn blank(out: &mut Vec<u8>, b: &[u8], from: usize, to: usize) {
    for &c in &b[from..to] {
        out.push(if c == b'\n' { b'\n' } else { b' ' });
    }
}

/// Push `[from, to)` unchanged.
fn blank_span(out: &mut Vec<u8>, b: &[u8], from: usize, to: usize) {
    out.extend_from_slice(&b[from..to]);
}

/// The byte after the opening quote of a raw string at `i`, with its hash
/// count, or `None` when `i` does not open one.
///
/// `r` and `br` only open a raw string when they do not continue an
/// identifier, which is what tells `br"x"` apart from the `r"` inside
/// `holder.expect("..`.
fn raw_string_start(b: &[u8], i: usize) -> Option<(usize, usize)> {
    let prev_ident = i > 0 && (b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_');
    if prev_ident {
        return None;
    }
    let mut k = if b[i] == b'r' {
        i + 1
    } else if b[i] == b'b' && i + 1 < b.len() && b[i + 1] == b'r' {
        i + 2
    } else {
        return None;
    };
    let hashes_from = k;
    while k < b.len() && b[k] == b'#' {
        k += 1;
    }
    if k < b.len() && b[k] == b'"' {
        Some((k + 1, k - hashes_from))
    } else {
        None
    }
}

/// One past the closing quote of the character literal at `i`, or `None` when
/// the `'` opens a lifetime instead.
///
/// The distinction is the whole point: treating `&'a str` as a literal
/// swallows to the next `'` and is exactly the blindness this file exists to
/// avoid.
fn char_literal_end(b: &[u8], i: usize) -> Option<usize> {
    let n = b.len();
    if i + 1 >= n {
        return None;
    }
    if b[i + 1] == b'\\' {
        // The escaped character sits at `i + 2`, so the search for the closing
        // quote starts past it; `'\''` is the case that needs this.
        let mut j = i + 3;
        while j < n && b[j] != b'\'' && b[j] != b'\n' {
            j += 1;
        }
        return (j < n && b[j] == b'\'').then_some(j + 1);
    }
    let len = match b[i + 1] {
        0x00..=0x7f => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        _ => 4,
    };
    (i + 1 + len < n && b[i + 1 + len] == b'\'').then_some(i + 2 + len)
}

/// Every `.rs` file under `dir`, recursively, as `(path relative to `dir`,
/// full path)`.
///
/// Recursive on purpose. `src/` is flat today, so a walk that stops at the top
/// gives the same answer and the `files.len() > 30` control cannot tell them
/// apart; the first `src/anything/mod.rs` added would exit this guard in
/// silence (issue #949). `tests/miri_ignore_convention.rs` already recurses,
/// and two of the three walks agreeing is not a rule.
fn rs_files_under(dir: &Path) -> Vec<(String, PathBuf)> {
    fn walk(dir: &Path, prefix: &str, out: &mut Vec<(String, PathBuf)>) {
        let mut entries: Vec<PathBuf> = fs::read_dir(dir)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()))
            .map(|e| e.expect("cannot read a directory entry").path())
            .collect();
        entries.sort();
        for path in entries {
            let name = path
                .file_name()
                .expect("a directory entry has a name")
                .to_string_lossy()
                .into_owned();
            let rel = if prefix.is_empty() {
                name.clone()
            } else {
                format!("{prefix}/{name}")
            };
            if path.is_dir() {
                walk(&path, &rel, out);
            } else if name.ends_with(".rs") {
                out.push((rel, path));
            }
        }
    }
    let mut out = Vec::new();
    walk(dir, "", &mut out);
    out.sort();
    out
}

fn files_with_real_unsafe() -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let files = rs_files_under(&src);
    assert!(
        files.len() > 30,
        "positive control failed: only {} files found under src/, so an empty \
         answer below would be an empty answer about nothing",
        files.len()
    );
    for (rel, path) in files {
        let text = fs::read_to_string(&path).expect("read source file");
        let code = code_only(&text);
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
/// Read as entries rather than as lines, because a `[features]` entry can span
/// lines and `jxl` does.
fn feature_table(manifest: &str) -> std::collections::BTreeMap<String, Vec<String>> {
    let mut out = std::collections::BTreeMap::new();
    let mut inside = false;
    let mut pending: Option<(String, String)> = None;
    for line in manifest.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') && pending.is_none() {
            inside = trimmed == "[features]";
            continue;
        }
        if !inside || trimmed.starts_with('#') {
            continue;
        }
        if let Some((name, rest)) = pending.take() {
            let joined = format!("{rest} {trimmed}");
            if joined.contains(']') {
                out.insert(name, enabled_features(&joined));
            } else {
                pending = Some((name, joined));
            }
            continue;
        }
        let Some((name, rest)) = trimmed.split_once('=') else {
            continue;
        };
        let (name, rest) = (name.trim(), rest.trim());
        if name.is_empty() || !rest.starts_with('[') {
            continue;
        }
        if rest.contains(']') {
            out.insert(name.to_owned(), enabled_features(rest));
        } else {
            pending = Some((name.to_owned(), rest.to_owned()));
        }
    }
    out
}

/// The feature names inside one `[..]` list, dropping `dep:x` and
/// `crate/feature` entries, which turn on a dependency rather than a feature
/// of this crate.
fn enabled_features(list: &str) -> Vec<String> {
    list.split('"')
        .skip(1)
        .step_by(2)
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
    let mut seen: BTreeSet<String> = BTreeSet::new();
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
    assert!(code_only("fn f() { unsafe { g() } }").contains("unsafe"));
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
    ] {
        // Collected rather than asserted row by row, so one run names every
        // literal form the scanner loses the file to instead of the first.
        if !code_only(code).contains("unsafe") {
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
            !code_only(prose).contains("unsafe"),
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
    let found: Vec<String> = rs_files_under(&root).into_iter().map(|(r, _)| r).collect();
    assert!(
        found.contains(&"fuzz_targets/fuzz_fits.rs".to_owned()),
        "the walk did not descend into `fuzz/fuzz_targets/`, so a module in a \
         subdirectory of `src/` would be invisible to this inventory and it \
         would report a clean tree. Found: {found:?}"
    );
}
