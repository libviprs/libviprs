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

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

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

/// Strip block comments, line comments and string literals, so the scan reads code
/// and not prose. A scanner that counts the sentence explaining a rule as a breach
/// of that rule is worse than no scanner.
fn code_only(src: &str) -> String {
    // Byte-wise on purpose: slicing `&src[i..=i + 1]` panics on a multi-byte char,
    // and `src/` has them (a `\u{d7}` in at least one comment).
    let b = src.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(b.len());
    let (mut i, mut in_str, mut in_line, mut in_block) = (0usize, false, false, 0usize);
    while i < b.len() {
        let opens = i + 1 < b.len() && b[i] == b'/' && b[i + 1] == b'*';
        let closes = i + 1 < b.len() && b[i] == b'*' && b[i + 1] == b'/';
        let line = i + 1 < b.len() && b[i] == b'/' && b[i + 1] == b'/';
        if in_line {
            if b[i] == b'\n' {
                in_line = false;
                out.push(b'\n');
            }
        } else if in_block > 0 {
            if closes {
                in_block -= 1;
                i += 2;
                continue;
            } else if opens {
                in_block += 1;
                i += 2;
                continue;
            }
        } else if in_str {
            if b[i] == b'\\' {
                i += 2;
                continue;
            }
            if b[i] == b'"' {
                in_str = false;
            }
        } else if line {
            in_line = true;
        } else if opens {
            in_block = 1;
            i += 2;
            continue;
        } else if b[i] == b'"' {
            in_str = true;
        } else {
            out.push(b[i]);
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
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
