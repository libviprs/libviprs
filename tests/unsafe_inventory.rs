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
use std::path::Path;

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

fn files_with_real_unsafe() -> BTreeSet<String> {
    let mut found = BTreeSet::new();
    for entry in fs::read_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("src")).expect("read src/")
    {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let text = fs::read_to_string(&path).expect("read source file");
        let code = code_only(&text);
        if code
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .any(|w| w == "unsafe")
        {
            found.insert(path.file_name().unwrap().to_string_lossy().into_owned());
        }
    }
    found
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

#[cfg_attr(miri, ignore)]
#[test]
fn the_crates_own_unsafe_stays_out_of_a_default_build() {
    // The claim that survives from merge-gate.yml's old paragraph: a default build,
    // and so a bare `cargo miri test`, reaches none of this crate's own `unsafe`.
    let manifest = fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"))
        .expect("read Cargo.toml");
    let default = manifest
        .lines()
        .find(|l| l.starts_with("default = "))
        .expect("Cargo.toml must declare a default feature list");
    for (file, _) in ALLOWED {
        let feature = file.trim_end_matches(".rs");
        assert!(
            !default.contains(feature),
            "`{feature}` is in the default feature list, so a default build now compiles \
             this crate's own `unsafe`. merge-gate.yml's paragraph on Miri coverage is no \
             longer true and must be rewritten (issue #897).\n  default = {default}"
        );
    }
}
