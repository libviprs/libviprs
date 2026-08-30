//! Source-masking and file-walking helpers shared by more than one
//! filesystem-scanning guard under `tests/` (issue #968).
//!
//! `mask_literals_and_comments` and `rs_files_under` each used to have two or
//! more separately-maintained copies. `tests/sample_kind_spine.rs` and
//! `tests/unsafe_inventory.rs` carried byte-for-byte identical masking
//! lexers (`mask_literals_and_comments` and `code_only`), and
//! `rs_files_under` was reimplemented in those same two files plus
//! `tests/ci_feature_coverage.rs`. Keeping copies "in step by hand" is
//! exactly the failure mode that produced issue #943: a `'"'` character
//! literal desynced one masking copy's string-tracking and not the other,
//! and the desynced copy silently lost about 2650 lines of `src/jp2k.rs` to
//! its scan while looking green. One copy cannot drift from another that
//! does not exist.
//!
//! `tests/miri_ignore_convention.rs` keeps its own file walker rather than
//! this one's `rs_files_under`: it accumulates several roots
//! (`SCANNED_DIRS`) into one caller-owned `Vec` through an out-parameter and
//! a `rel_prefix`, where every caller here only ever walks a single root and
//! wants the `Vec` handed back in one call. Reconciling the two shapes would
//! mean changing that file's multi-root call site rather than a mechanical
//! extraction, so it is left as is.

#![allow(dead_code)]

/// Replace comments and the *contents* of every literal with spaces, keeping
/// each byte where it was so line numbers survive.
///
/// Deliberately not a Rust parser, and deliberately more than a line-oriented
/// strip. It tracks strings, byte strings, raw strings of any hash depth, and
/// character literals, because each of those can carry a `//` or a `"` that a
/// simpler scan reads as a delimiter and then loses the rest of the file to.
/// That is not hypothetical: `tests/unsafe_inventory.rs` lost 2650 lines of
/// `src/jp2k.rs` to a single `'"'` (issue #943), and `src/svg.rs` puts
/// `http://www.w3.org/2000/svg` inside a raw string, which a scanner that does
/// not know `br##"` reads as a line comment.
///
/// A `'` is a character literal only when it really closes one. `&'a str` is a
/// lifetime, and swallowing from there to the next `'` is the same
/// desynchronisation in a new dress, so [`char_literal_end`] decides.
pub fn mask_literals_and_comments(src: &str) -> String {
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
    // `c"..."` (a non-raw C-string) needs nothing here: it falls through to
    // the plain `"` branch a byte later and is masked correctly by luck,
    // since there is no prefix-specific escaping difference from an
    // ordinary string. `cr"..."` does need to be here, the same as `br"..."`
    // beside it, or a literal backslash inside one is escape-processed and
    // desynchronises the mask exactly the way #943 already happened once
    // (issue #940's panel).
    let mut k = if b[i] == b'r' {
        i + 1
    } else if (b[i] == b'b' || b[i] == b'c') && i + 1 < b.len() && b[i + 1] == b'r' {
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
///
/// A new test calling this (or any other filesystem-touching helper added to
/// this module) must self-annotate `#[cfg_attr(miri, ignore)]`.
/// `tests/miri_ignore_convention.rs`'s own detector only follows a call graph
/// within one file, so it cannot see a test reach the filesystem through a
/// call into `tests/common/`: nothing will catch a missing annotation on a
/// new caller the way it would for filesystem access written inline (issue
/// #940, found while extracting this function out of the files that used to
/// carry the annotated calls it could still see).
pub fn rs_files_under(dir: &std::path::Path) -> Vec<(String, std::path::PathBuf)> {
    fn walk(dir: &std::path::Path, prefix: &str, out: &mut Vec<(String, std::path::PathBuf)>) {
        let mut entries: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
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
