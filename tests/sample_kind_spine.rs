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
//! # Why there is a countdown rather than a flat zero, for now
//!
//! One code site is left on `main` and it is not in a file this lane owns.
//! `src/conversion.rs`'s went with the carriers lane, and it went by being
//! *deleted* rather than converted: the site was `addalpha`'s
//! `bytes_per_channel() == 1` alpha ceiling, and issue #861's fix replaced
//! the width rule with the interpretation's max alpha, so there is no
//! comparison left to key on.
//!
//! It is named in [`REMAINING`] with the lane that clears it. The
//! assertion is set equality in both directions, the way
//! `tests/ci_feature_coverage.rs` holds the `Makefile` and the CI table
//! together: a new site anywhere fails, and an entry here that has already been
//! cleared **also** fails, so the list can only shrink and cannot go stale.
//! When it is empty this is the flat zero #607 wants, and #607 closes.

/// The width comparisons still on `main`, each with the lane that owns the file.
///
/// Shrink this, never grow it. An entry whose site is gone fails the test, so
/// clearing a site means deleting its line here in the same PR.
const REMAINING: &[(&str, &str)] = &[];

/// Strip `//` line comments and `/* */` block comments, replacing them with
/// spaces so byte offsets and line numbers survive.
///
/// Deliberately not a Rust parser. It does track string and char literals,
/// because a `//` inside a string is not a comment and dropping the rest of
/// that line would hide real code after it. Raw strings are not handled and
/// do not need to be: nothing in `src/` puts a width comparison in one, and
/// the test below proves the stripper on both a code case and a comment case
/// rather than asserting it works.
fn strip_comments(src: &str) -> String {
    // Over bytes rather than over `char`s, because the sources carry non-ASCII
    // prose (a `±` in one comment) and slicing a `&str` mid-codepoint panics.
    // Every byte of a multi-byte codepoint is >= 0x80, so it can never be one
    // of the ASCII delimiters below and is copied through unchanged.
    let bytes = src.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(src.len());
    let mut i = 0;
    let mut depth = 0usize;
    while i < bytes.len() {
        let two = &bytes[i..(i + 2).min(bytes.len())];
        if depth > 0 {
            if two == b"/*" {
                depth += 1;
                out.extend_from_slice(b"  ");
                i += 2;
                continue;
            }
            if two == b"*/" {
                depth -= 1;
                out.extend_from_slice(b"  ");
                i += 2;
                continue;
            }
            out.push(if bytes[i] == b'\n' { b'\n' } else { b' ' });
            i += 1;
            continue;
        }
        if two == b"/*" {
            depth = 1;
            out.extend_from_slice(b"  ");
            i += 2;
            continue;
        }
        if two == b"//" {
            while i < bytes.len() && bytes[i] != b'\n' {
                out.push(b' ');
                i += 1;
            }
            continue;
        }
        if bytes[i] == b'"' || bytes[i] == b'\'' {
            let quote = bytes[i];
            out.push(quote);
            i += 1;
            while i < bytes.len() {
                if bytes[i] == b'\\' && i + 1 < bytes.len() {
                    out.extend_from_slice(&bytes[i..i + 2]);
                    i += 2;
                    continue;
                }
                let c = bytes[i];
                out.push(c);
                i += 1;
                if c == quote || c == b'\n' {
                    break;
                }
            }
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8(out).expect("only ASCII delimiters are rewritten, and each as one byte")
}

/// Every line of `src` that compares a byte width, as `(line number, text)`.
///
/// Both orders, because `bytes_per_channel() == 1` and `1 == bpc()` are the
/// same mistake, and `!=` as well as `==`.
fn width_comparisons(src: &str) -> Vec<(usize, String)> {
    const NEEDLE: &str = "bytes_per_channel()";
    strip_comments(src)
        .lines()
        .enumerate()
        .filter(|(_, line)| {
            let Some(at) = line.find(NEEDLE) else {
                return false;
            };
            let after = line[at + NEEDLE.len()..].trim_start();
            // Walk back over the receiver expression, so `4 == x.f().g()`
            // is seen as a comparison and not as whatever character the
            // receiver happens to end on.
            let before = line[..at]
                .trim_end_matches(|c: char| c.is_alphanumeric() || "_.()&*?: <>".contains(c))
                .trim_end();
            after.starts_with("==")
                || after.starts_with("!=")
                || before.ends_with("==")
                || before.ends_with("!=")
        })
        .map(|(n, line)| (n + 1, line.trim().to_string()))
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
}

/**
 * Tests that no file under `src/` compares a byte width, except the ones
 * [`REMAINING`] names, and that every file [`REMAINING`] names still has one.
 * A byte width is not a sample kind: 4 is `f32` today and would be `u32`
 * under #517 and `i32` under #516, so a width comparison is a question with
 * three right answers and one arm. `#[non_exhaustive]` on `SampleKind` turns
 * a `match` into a compile error and does nothing at all to a comparison,
 * which is why this is a scan and not a lint (issue #607 step (e)).
 * Works by stripping comments from each file and looking for the comparison
 * in either order with either operator. Set equality in both directions, so
 * the list can only shrink.
 * Input: every Rust file under `src` -> Output: comparisons only in the
 * named files, and a comparison in each of them.
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
    let named: Vec<&str> = REMAINING.iter().map(|&(f, _)| f).collect();
    let mut still_there: Vec<&str> = Vec::new();

    for path in src_files() {
        let rel = format!(
            "src/{}",
            path.file_name()
                .expect("a file has a name")
                .to_string_lossy()
        );
        let src = std::fs::read_to_string(&path).expect("a source file is readable");
        let hits = width_comparisons(&src);
        if hits.is_empty() {
            continue;
        }
        let owner = REMAINING.iter().find(|&&(f, _)| f == rel);
        let Some(&(_, lane)) = owner else {
            let lines: Vec<String> = hits
                .iter()
                .map(|(n, text)| format!("{rel}:{n}: {text}"))
                .collect();
            panic!(
                "a byte width is compared in {rel}, which is not on the \
                 countdown. Dispatch on PixelFormat::kind() instead: a width \
                 cannot separate f32 from the u32 and i32 carriers of #517 \
                 and #516, and nothing in the type system catches this \
                 (issue #607).\n{}",
                lines.join("\n")
            );
        };
        let _ = lane;
        still_there.push(named.iter().find(|&&f| f == rel).expect("just matched"));
    }

    still_there.sort_unstable();
    let mut expected = named.clone();
    expected.sort_unstable();
    assert_eq!(
        still_there, expected,
        "the countdown is stale. Every file listed in REMAINING must still \
         have a width comparison; one that has been cleared has to lose its \
         line here in the same change, so the list cannot rot into an \
         allowlist. When it is empty, issue #607 closes."
    );
}
