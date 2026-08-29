//! The gate issue #607 step (e) asks for: a byte width must not stand in for a
//! sample kind anywhere in `src/`.
//!
//! # What this is for
//!
//! `PixelFormat::bytes_per_channel()` answers 4 for `f32` and would answer 4
//! for the `u32` carrier of #517 and the `i32` one of #516. So
//! `bytes_per_channel() == 4` is a question with three different right answers
//! and one arm, and every site that asked it was silently wrong for two of the
//! three. #748 sorted 31 such sites by what they actually did and found 16 that
//! were definitely wrong and silent, one of which wrote a `float` tag into a
//! `.v` file (#841).
//!
//! Those 16 are fixed. This is what stops a seventeenth, because
//! `#[non_exhaustive]` on `SampleKind` only turns a *`match`* into a compile
//! error, and a width **comparison** is not a match. Nothing in the type system
//! catches `if fmt.bytes_per_channel() == 1`, which is why #607 asked for a
//! grep rather than for a lint.
//!
//! It lives here rather than in `.github/workflows/` so it runs under
//! `cargo test`, which is both the local gate and CI's Test job, and so it can
//! parse rather than pattern-match. A shell `grep` cannot tell a comment from
//! code, and the very first thing it would have failed on is `pixel.rs`'s
//! comment explaining why `canonical()` does **not** take the width shortcut,
//! which is prose arguing *for* this rule.
//!
//! # The four shapes, because a scan that reads one of them is not a gate
//!
//! #607 was closed on a scanner whose needle was `bytes_per_channel()` sitting
//! next to `==` or `!=` **on one line**. Three of the four shapes a real site
//! takes got past it, each one measured as a plant in `src/draw.rs` that
//! `cargo fmt --check` and `cargo check --lib` both accepted (issue #942):
//!
//! * the comparison **rustfmt-wrapped**, with the operator opening the
//!   continuation line, which is what any joined line over 100 characters
//!   becomes under this repo's own fmt gate;
//! * a **match head**, `match bytes_per_channel() { 1 => .., 2 => .., _ => .. }`,
//!   which is the shape the epic's own headline number counted ("width-keyed
//!   match heads 12 -> 7 -> 2");
//! * an **ordering operator**, `> 2`, which is the same question with a coarser
//!   arm.
//!
//! It also never read the `SampleKind::bytes()` spelling at all, and that is
//! where three of the six live sites turned out to be.
//!
//! So the scan masks comments and literals, tokenises what is left, and asks
//! about tokens rather than about lines. Where rustfmt broke the line stops
//! mattering, `match` and `matches!` heads are read as the comparisons they
//! are, and `buf[..kind.bytes()]` still is not one, because the walk back over
//! the receiver stops at the `[` instead of running on to whatever operator
//! comes next.
//!
//! # The two lists, and why neither can rot
//!
//! [`REMAINING`] is the countdown: files with a site somebody still owes a
//! conversion, keyed by file because it tracks a lane's outstanding work. It
//! only ever shrinks, and an entry whose site is gone fails, so it cannot turn
//! into an allowlist. It has one entry, `src/fits.rs`, and it earned it: the
//! match head there is not just a width standing in for a kind, it is a
//! *wrong* answer for all four carriers #516 and #517 added, measured against
//! vips 8.18.6 on both sides (issue #957). Converting it without adding the
//! 32-bit integer carrier would freeze that answer in a shape that reads as
//! done.
//!
//! [`DELIBERATE`] is the other kind: sites where the byte width **is** the
//! thing under test, keyed by their exact text so a second comparison in the
//! same file is still a failure. There is one, and it is
//! `src/composite.rs`'s control that `SampleKind::promote` agrees with the old
//! "wider width wins" rule on the three kinds a `PixelFormat` carries.
//!
//! Being non-empty is what makes the headline assertion honest. It is set
//! equality over the sites found under `src/`, and a scanner that has stopped
//! working finds nothing; a one-element expectation fails on the empty set
//! exactly as it fails on a new site, so there is no vacuous state left to
//! guard. That is the shape issue #939 arrived at for the `sha2` floor, and it
//! beats a positive control because nobody has to remember it.

use std::collections::BTreeSet;

/// The width comparisons still on `main`, each with the lane that owns the file.
///
/// Shrink this, never grow it. An entry whose site is gone fails the test, so
/// clearing a site means deleting its line here in the same PR.
const REMAINING: &[(&str, &str)] = &[(
    "src/fits.rs",
    "issue #957: Carrier::for_format keys the FITS carrier on the width, and \
     converting it without adding the 32-bit integer carrier would freeze a \
     measured-wrong answer. vips 8.18.6 writes an `int` image as BITPIX 32 \
     with BZERO 2147483648; libviprs writes BITPIX -32 and reinterprets the \
     integer bytes as f32, and it writes an `Int8` -5 as 251 where vips \
     saturates to 0",
)];

/// Sites where the byte width **is** the thing under test, with the reason.
///
/// Pinned by their text rather than by their file, so a second comparison in
/// the same file is still a failure and this cannot rot into a per-file
/// allowlist the way [`REMAINING`] deliberately can (that one only ever
/// shrinks).
///
/// It also does the job a positive control would. The headline assertion is
/// "the set of width comparisons under `src/` is exactly this", and a scanner
/// that has stopped working produces the empty set. A one-element expectation
/// fails on the empty set exactly as it fails on a new site, so there is no
/// vacuous state left to guard, which is the shape issue #939 arrived at for
/// the `sha2` floor.
const DELIBERATE: &[(&str, &str, &str)] = &[(
    "src/composite.rs:1682",
    "let by_width = if a.bytes() >= b.bytes() { a } else { b };",
    "the old \"wider width wins\" rule, kept as the control that \
     SampleKind::promote agrees with it on the three kinds a PixelFormat \
     carries. Converting this one would delete the comparison the test exists \
     to make",
)];

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
fn mask_literals_and_comments(src: &str) -> String {
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

/// Multi-character operators, longest first so `..=` is not read as `..` then
/// `=` and `<<` is not read as two comparisons.
const OPERATORS: &[&str] = &[
    "..=", "...", "<<=", ">>=", "==", "!=", "<=", ">=", "&&", "||", "::", "..", "->", "=>", "<<",
    ">>", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=",
];

/// The comparison operators a byte width can be asked about. Ordering counts:
/// `bytes_per_channel() > 2` is the same question as `== 4` with a coarser arm,
/// and `src/composite.rs` really did key an output depth on `>=` (issue #942).
const COMPARISONS: &[&str] = &["==", "!=", "<=", ">=", "<", ">"];

/// A crude token stream over masked source, as `(byte offset, text)`.
///
/// Identifiers and numbers come out whole, the operators above come out whole,
/// and everything else is one character. That is enough to tell
/// `a.bytes() >= b.bytes()` from `buf[..kind.bytes()]` without being a parser,
/// and it is what makes the scan indifferent to where rustfmt broke the line.
fn tokens(masked: &str) -> Vec<(usize, &str)> {
    let b = masked.as_bytes();
    let n = b.len();
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < n {
        if b[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }
        if b[i].is_ascii_alphanumeric() || b[i] == b'_' {
            let start = i;
            while i < n && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            out.push((start, &masked[start..i]));
            continue;
        }
        if let Some(op) = OPERATORS.iter().find(|op| masked[i..].starts_with(**op)) {
            out.push((i, *op));
            i += op.len();
            continue;
        }
        let len = match b[i] {
            0x00..=0x7f => 1,
            0xc0..=0xdf => 2,
            0xe0..=0xef => 3,
            _ => 4,
        };
        out.push((i, &masked[i..i + len]));
        i += len;
    }
    out
}

/// Whether a token is an identifier or a number rather than punctuation.
fn is_word(tok: &str) -> bool {
    tok.as_bytes()
        .first()
        .is_some_and(|c| c.is_ascii_alphanumeric() || *c == b'_')
}

/// The index of the first token of the receiver expression whose method is
/// being called, given the index of the `.` or `::` in front of the method
/// name.
///
/// Walks back over identifiers, path separators, `?`, and balanced `(..)` and
/// `[..]` groups, and stops at anything else. Stopping is what keeps
/// `a == 1 && read(data, kind.bytes())` from reading as a width comparison:
/// the walk halts at the `,` and never reaches the `==`.
fn receiver_start(toks: &[(usize, &str)], dot: usize) -> usize {
    let mut k = dot;
    loop {
        if k == 0 {
            return 0;
        }
        let t = toks[k - 1].1;
        if t == ")" || t == "]" {
            let (open, close) = if t == ")" { ("(", ")") } else { ("[", "]") };
            let mut depth = 0usize;
            let mut m = k - 1;
            loop {
                if toks[m].1 == close {
                    depth += 1;
                } else if toks[m].1 == open {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                if m == 0 {
                    return 0;
                }
                m -= 1;
            }
            k = m;
            continue;
        }
        if t == "." || t == "::" || t == "?" || is_word(t) {
            k -= 1;
            continue;
        }
        return k;
    }
}

/// The byte ranges holding the scrutinee of a `match` or the first argument of
/// a `matches!`.
///
/// A match head is a width comparison with one arm per answer, and it is the
/// shape this epic's own metric counted ("width-keyed match heads 12 -> 7 ->
/// 2") while the gate that closed #607 could not see one at all.
fn head_ranges(toks: &[(usize, &str)]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for (i, (_, t)) in toks.iter().enumerate() {
        let (from, stops) = if *t == "match" {
            (i + 1, ["{"].as_slice())
        } else if *t == "matches"
            && toks.get(i + 1).map(|t| t.1) == Some("!")
            && toks.get(i + 2).map(|t| t.1) == Some("(")
        {
            (i + 3, [",", ")"].as_slice())
        } else {
            continue;
        };
        let Some(&(start, _)) = toks.get(from) else {
            continue;
        };
        let mut depth = 0usize;
        for (off, tok) in &toks[from..] {
            if depth == 0 && stops.contains(tok) {
                out.push((start, *off));
                break;
            }
            match *tok {
                "(" | "[" | "{" => depth += 1,
                ")" | "]" | "}" => depth = depth.saturating_sub(1),
                _ => {}
            }
        }
    }
    out
}

/// Every line of `src` where a byte width is asked to stand in for a sample
/// kind, as `(line number, the masked line, whitespace-collapsed)`.
///
/// Four shapes, all of them measured on plants in `src/draw.rs` that
/// `cargo fmt --check` and `cargo check --lib` both accepted (issue #942):
/// the comparison on one line, the same comparison rustfmt-wrapped, a match
/// head, and an ordering operator. Both operand orders, both equality
/// operators, all four ordering operators, and both spellings of the
/// accessor: `PixelFormat::bytes_per_channel()` and `SampleKind::bytes()`.
///
/// # Where it under-approximates
///
/// A width read into a local and compared on the next line
/// (`let bpc = kind.bytes(); if bpc == 1`) is invisible, and so is a width
/// compared through a helper in another file. Both need a scan that follows
/// values rather than one that reads tokens, which is a different kind of
/// program from this one. `the_documented_blind_spots_are_still_blind` pins
/// them so the list cannot grow in silence.
fn width_comparisons(src: &str) -> Vec<(usize, String)> {
    const NEEDLES: [&str; 2] = ["bytes_per_channel", "bytes"];
    let masked = mask_literals_and_comments(src);
    let toks = tokens(&masked);
    let heads = head_ranges(&toks);
    let mut lines: Vec<usize> = Vec::new();
    for (i, (off, text)) in toks.iter().enumerate() {
        if !NEEDLES.contains(text) {
            continue;
        }
        // A method call, so the accessor's own definition and the word in a
        // field name are not hits.
        if i == 0 || (toks[i - 1].1 != "." && toks[i - 1].1 != "::") {
            continue;
        }
        if toks.get(i + 1).map(|t| t.1) != Some("(") || toks.get(i + 2).map(|t| t.1) != Some(")") {
            continue;
        }
        let after = toks.get(i + 3).is_some_and(|t| COMPARISONS.contains(&t.1));
        let start = receiver_start(&toks, i - 1);
        let before = start > 0 && COMPARISONS.contains(&toks[start - 1].1);
        let in_head = heads.iter().any(|&(a, b)| *off >= a && *off < b);
        if after || before || in_head {
            lines.push(masked[..*off].bytes().filter(|c| *c == b'\n').count() + 1);
        }
    }
    lines.sort_unstable();
    lines.dedup();
    lines
        .into_iter()
        .map(|n| {
            let text = masked.lines().nth(n - 1).unwrap_or_default();
            (n, text.split_whitespace().collect::<Vec<_>>().join(" "))
        })
        .collect()
}

fn src_files() -> Vec<std::path::PathBuf> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .expect("src/ is readable")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "rs"))
        .collect();
    files.sort();
    assert!(
        files.len() > 30,
        "positive control failed: only {} files found under src/, so a zero \
         below would be a zero of nothing",
        files.len()
    );
    files
}

/**
 * Tests that the comment stripper this gate is built on actually
 * distinguishes code from prose, on both a case that must be seen and a case
 * that must not.
 * Works by running the scanner over two literal sources rather than over the
 * crate, so the gate is proved before it is trusted. Without the second case
 * this test would fail on `pixel.rs`'s own comment arguing for the rule, and
 * an allowlist entry for that would freeze the sentence in place as if it
 * were a permitted exception.
 * Input: a width comparison in code, then the same text inside each comment
 * form -> Output: found, then not found.
 */
#[test]
fn the_width_scanner_sees_code_and_not_comments() {
    let code = "fn f(x: PixelFormat) -> bool { x.bytes_per_channel() == 1 }";
    assert_eq!(
        width_comparisons(code).len(),
        1,
        "a width comparison in code must be found"
    );

    for cloaked in [
        "// x.bytes_per_channel() == 1\nfn f() {}",
        "/// x.bytes_per_channel() == 1\nfn f() {}",
        "/* x.bytes_per_channel() == 1 */\nfn f() {}",
        "/**\n * x.bytes_per_channel() == 1\n */\nfn f() {}",
    ] {
        assert_eq!(
            width_comparisons(cloaked),
            Vec::<(usize, String)>::new(),
            "the same text inside a comment must not be found: {cloaked:?}"
        );
    }

    // And a `//` inside a string is not a comment, so code after it still
    // counts. This is the case a line-oriented stripper gets wrong.
    let in_string = r#"fn f(x: PixelFormat) -> bool { let _ = "//"; x.bytes_per_channel() == 2 }"#;
    assert_eq!(
        width_comparisons(in_string).len(),
        1,
        "a `//` inside a string literal must not blind the scanner"
    );

    // Both orders and both operators.
    for spelling in [
        "if x.bytes_per_channel() == 4 {}",
        "if x.bytes_per_channel() != 4 {}",
        "if 4 == x.bytes_per_channel() {}",
    ] {
        assert_eq!(
            width_comparisons(spelling).len(),
            1,
            "unseen spelling: {spelling}"
        );
    }

    // Each literal form that can desynchronise the mask and swallow the rest
    // of the file, with the crippling of the mask each row catches, measured
    // by running the mask with that half removed:
    //
    // | row | catches |
    // |---|---|
    // | `'"'` | not tracking character literals at all, which is what cost
    //   `tests/unsafe_inventory.rs` 2650 lines of `src/jp2k.rs` (issue #943) |
    // | a lifetime on the comparison's own line | treating every `'` as
    //   opening a literal, which swallows from `&'static` to the next quote |
    // | a raw string holding an unmatched `"` | not knowing `r#"`, which is
    //   how `src/svg.rs`'s `http://www.w3.org/2000/svg` reads as a comment |
    //
    // The escaped-quote row is inert as a mutation and kept anyway: under the
    // rule below a mishandled `'\''` degrades into a lifetime rather than
    // into a swallowed span, so nothing is lost, but the literal is common
    // enough in this tree to be worth pinning.
    for (label, src) in [
        (
            "a char literal holding a quote",
            "fn f(s: &str, x: PixelFormat) -> bool {\n    let _ = s.split_once('\"');\n    \
             x.bytes_per_channel() == 1\n}\n",
        ),
        (
            "a char literal whose escape is a quote, beside one that holds a quote",
            "fn f(s: &str, x: PixelFormat) -> bool {\n    \
             let _ = (s.split_once('\\''), s.split_once('\"'));\n    \
             x.bytes_per_channel() == 1\n}\n",
        ),
        (
            "a raw string holding an unmatched quote",
            "fn f(x: PixelFormat) -> bool {\n    let _ = r#\"an unmatched \" quote\"#;\n    \
             x.bytes_per_channel() == 1\n}\n",
        ),
        (
            "a lifetime on the line the comparison is on",
            "fn f(x: PixelFormat) -> bool { let s: &'static str = \"n\"; \
             x.bytes_per_channel() == 1 }\n",
        ),
    ] {
        assert_eq!(
            width_comparisons(src).len(),
            1,
            "the scanner lost the rest of the file to {label}:\n{src}"
        );
    }

    // And the contents of a literal are prose, however they are spelled.
    for cloaked in [
        r#"fn f() { let _ = "x.bytes_per_channel() == 1"; }"#,
        r###"fn f() { let _ = r#"x.bytes_per_channel() == 1"#; }"###,
    ] {
        assert_eq!(
            width_comparisons(cloaked),
            Vec::<(usize, String)>::new(),
            "the same text inside a string literal must not be found: {cloaked:?}"
        );
    }
}

/**
 * Tests that the shapes this scanner cannot see are still the two the module
 * doc names, so the list of blind spots cannot grow in silence.
 * Works by running the scanner over one source per documented blind spot and
 * asserting it finds nothing. Both need a scan that follows values rather
 * than one that reads tokens, which is a different kind of program from this
 * one, so they are recorded rather than closed. Closing one turns this test
 * red, which is the moment the module doc needs the paragraph rewritten.
 * Input: a width read into a local, and a width passed to a helper ->
 * Output: no hits, for now.
 */
#[test]
fn the_documented_blind_spots_are_still_blind() {
    for (label, src) in [
        (
            "a width read into a local and compared on the next line",
            "fn f(kind: SampleKind) -> bool {\n    let bpc = kind.bytes();\n    bpc == 1\n}\n",
        ),
        (
            "a width handed to a helper that does the comparing",
            "fn f(kind: SampleKind) -> bool {\n    is_one_byte(kind.bytes())\n}\n",
        ),
    ] {
        assert_eq!(
            width_comparisons(src),
            Vec::<(usize, String)>::new(),
            "the scanner now sees `{label}`, which the module doc still lists \
             as a blind spot. Rewrite the paragraph and move this row into \
             MUST_BE_FOUND."
        );
    }
}

/**
 * Tests that no file under `src/` compares a byte width, except the ones
 * [`REMAINING`] names and the ones [`DELIBERATE`] explains, and that every
 * entry in both is still there.
 * A byte width is not a sample kind: 4 is `f32`, `u32` and `i32` at once, so
 * a width comparison is a question with three right answers and one arm.
 * `#[non_exhaustive]` on `SampleKind` turns a *`match`* into a compile error
 * and does nothing at all to a comparison, which is why this is a scan and
 * not a lint (issue #607 step (e)).
 * Works by masking comments and literals in each file, tokenising what is
 * left, and reading four shapes: the comparison on one line, the same one
 * rustfmt-wrapped, a match head, and an ordering operator, in both operand
 * orders and under both spellings of the accessor. The gate this replaces
 * saw only the first (issue #942).
 * Asserted as set equality against `DELIBERATE`, which is non-empty, so a
 * scanner that has stopped working fails here rather than passing on nothing.
 * Input: every Rust file under `src` -> Output: exactly the named sites.
 *
 * Carries `#[cfg_attr(miri, ignore)]` because it walks `src/` and reads every
 * file, which Miri's isolation layer refuses. Miri aborts the whole run on
 * the first refused operation rather than failing one test, so an
 * unannotated filesystem test takes the undefined-behaviour gate down and
 * reports as "Miri failed" (issue #652). It is recorded in
 * `tests/miri_fs_test_inventory.txt` too, which is what stops the annotation
 * being deleted unnoticed.
 */
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn a_byte_width_is_never_compared_outside_the_named_countdown() {
    let mut found: BTreeSet<(String, String)> = BTreeSet::new();
    for path in src_files() {
        let rel = format!(
            "src/{}",
            path.file_name()
                .expect("a file has a name")
                .to_string_lossy()
        );
        let src = std::fs::read_to_string(&path).expect("a source file is readable");
        for (n, text) in width_comparisons(&src) {
            found.insert((format!("{rel}:{n}"), text));
        }
    }

    // A site in a file the countdown names is explained by that entry, wherever
    // in the file it sits; the countdown is per file because it tracks a lane's
    // outstanding work rather than one line.
    let counting_down: BTreeSet<&str> = REMAINING.iter().map(|&(f, _)| f).collect();
    let mut expected: BTreeSet<(String, String)> = DELIBERATE
        .iter()
        .map(|&(site, text, _)| (site.to_owned(), text.to_owned()))
        .collect();
    expected.extend(
        found
            .iter()
            .filter(|(site, _)| {
                counting_down
                    .iter()
                    .any(|f| site.starts_with(&format!("{f}:")))
            })
            .cloned(),
    );

    assert_eq!(
        found,
        expected,
        "the set of byte-width comparisons under src/ has moved.\n\
         A site that is here and not in the lists below is a new one: dispatch \
         on PixelFormat::kind() or SampleKind instead, because a width cannot \
         separate f32 from the u32 and i32 carriers of #517 and #516 and \
         nothing in the type system catches it (issue #607).\n\
         A site in the lists and not here has been cleared, so delete its \
         line in the same change.\n\
         New: {:?}\nGone: {:?}",
        found.difference(&expected).collect::<Vec<_>>(),
        expected.difference(&found).collect::<Vec<_>>(),
    );

    // The countdown itself: a file it names must still have a site, so an
    // entry cannot outlive the work it tracks.
    for &(file, lane) in REMAINING {
        assert!(
            found
                .iter()
                .any(|(site, _)| site.starts_with(&format!("{file}:"))),
            "the countdown is stale: {file} is listed against {lane} and has \
             no width comparison left. Delete the line. When REMAINING is \
             empty, issue #607 closes."
        );
    }
}

/// The shapes a real width comparison arrives in, each one measured on a plant
/// in `src/draw.rs` that `cargo fmt --check` and `cargo check --lib` both
/// accepted (issue #942).
///
/// Every row is a source that *must* be found. Three of the four were green
/// under the scanner #607 was closed on, and the third of them is the one that
/// matters: the epic's own headline number for this work was "width-keyed match
/// heads 12 -> 7 -> 2", and the gate that closed it could not see a match head.
const MUST_BE_FOUND: &[(&str, &str)] = &[
    (
        "one line, the only shape the #607 scanner could see",
        "fn f(fmt: PixelFormat) -> bool {\n    fmt.bytes_per_channel() == 4\n}\n",
    ),
    (
        "rustfmt-wrapped, the operator opening the continuation line",
        "fn f(fmt: PixelFormat) -> bool {\n    \
         a_receiver_long_enough_to_make_rustfmt_wrap.bytes_per_channel()\n        == 4\n}\n",
    ),
    (
        "rustfmt-wrapped the other way round, the constant first",
        "fn f(fmt: PixelFormat) -> bool {\n    A_CONSTANT_WITH_A_LONG_NAME\n        \
         == fmt.bytes_per_channel()\n}\n",
    ),
    (
        "a match head, the shape the epic's own metric counted",
        "fn f(fmt: PixelFormat) -> Carrier {\n    match fmt.bytes_per_channel() {\n        \
         1 => Carrier::U8,\n        2 => Carrier::U16,\n        _ => Carrier::F32,\n    }\n}\n",
    ),
    (
        "a match head on the SampleKind spelling",
        "fn f(kind: SampleKind) -> u8 {\n    match kind.bytes() {\n        \
         1 => 1,\n        _ => 2,\n    }\n}\n",
    ),
    (
        "an ordering operator, which is the same question with a coarser arm",
        "fn f(fmt: PixelFormat) -> bool {\n    fmt.bytes_per_channel() > 2\n}\n",
    ),
    (
        "an ordering operator on the receiver's right",
        "fn f(a: SampleKind, b: SampleKind) -> SampleKind {\n    \
         if a.bytes() >= b.bytes() { a } else { b }\n}\n",
    ),
    (
        "the SampleKind::bytes() synonym, which the #607 scanner never read",
        "fn f(kind: SampleKind) -> f64 {\n    if kind.bytes() == 1 { 1.0 } else { 257.0 }\n}\n",
    ),
    (
        "matches!, which is a match head wearing a macro",
        "fn f(fmt: PixelFormat) -> bool {\n    matches!(fmt.bytes_per_channel(), 1 | 2)\n}\n",
    ),
];

/// Uses of the same accessors that are not a width standing in for a kind, so
/// the scanner must leave them alone.
///
/// The `assert_eq!` rows are a decision rather than an oversight: `src/pixel.rs`
/// pins `SampleKind::bytes()` against literal widths twenty times over, and
/// `kind().bytes() == bytes_per_channel()` is the assertion that the two
/// accessors agree. Those are the accessor under test, not a dispatch on it.
const MUST_NOT_BE_FOUND: &[(&str, &str)] = &[
    (
        "a stride",
        "fn f(kind: SampleKind) -> usize {\n    i * bands * kind.bytes()\n}\n",
    ),
    (
        "a buffer size",
        "fn f(kind: SampleKind) -> Vec<u8> {\n    vec![0u8; 2 * kind.bytes()]\n}\n",
    ),
    (
        "a slice bound",
        "fn f(kind: SampleKind) -> bool {\n    buf[..kind.bytes()].iter().all(|&b| b == 0)\n}\n",
    ),
    (
        "a plain binding",
        "fn f(kind: SampleKind) -> usize {\n    let bpc = kind.bytes();\n    bpc\n}\n",
    ),
    (
        "an assert pinning the accessor itself",
        "fn f(kind: SampleKind) {\n    assert_eq!(kind.bytes(), 4, \"four-byte kind\");\n}\n",
    ),
    (
        "an argument that happens to sit beside a comparison",
        "fn f(a: usize, kind: SampleKind) -> bool {\n    a == 1 && read(data, kind.bytes())\n}\n",
    ),
];

/**
 * Tests that the width scanner sees every shape a width comparison actually
 * arrives in, and not the uses that are a stride or a buffer size.
 * Works by running `width_comparisons` over one literal source per shape,
 * rather than over the crate, so the scanner is proved before the countdown
 * below is believed. Three of the shapes were measured green under the
 * scanner issue #607 was closed on, each one `cargo fmt --check` and
 * `cargo check --lib` clean as a plant in `src/draw.rs`: a wrapped line, a
 * match head and an ordering operator, plus the `SampleKind::bytes()`
 * spelling it never read at all (issue #942).
 * Input: one source per shape -> Output: found for every row of
 * `MUST_BE_FOUND`, not found for every row of `MUST_NOT_BE_FOUND`.
 */
#[test]
fn the_width_scanner_sees_every_shape_a_real_site_takes() {
    // Collected rather than asserted row by row, so one run names every shape
    // the scanner cannot see instead of stopping at the first.
    let mut wrong: Vec<String> = Vec::new();
    for (label, src) in MUST_BE_FOUND {
        let hits = width_comparisons(src).len();
        if hits != 1 {
            wrong.push(format!("MISSED ({label}), {hits} hits:\n{src}"));
        }
    }
    for (label, src) in MUST_NOT_BE_FOUND {
        let hits = width_comparisons(src);
        if !hits.is_empty() {
            wrong.push(format!("FALSE POSITIVE ({label}), {hits:?}:\n{src}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "the width scanner is {} rows wrong out of {}:\n{}",
        wrong.len(),
        MUST_BE_FOUND.len() + MUST_NOT_BE_FOUND.len(),
        wrong.join("\n")
    );
}
