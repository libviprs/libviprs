//! Guards that every checked-in oracle capture is JSON a strict parser reads
//! (issue #674).
//!
//! The captures under `oracle-captures/` are written by Python, and
//! `json.dump` spells a non-finite float as a bare `NaN`, `Infinity` or
//! `-Infinity` by default. Python's own `json.load` reads those back as a
//! documented non-standard extension, so a capture round-trips perfectly on
//! the machine that wrote it and is rejected WHOLE by `serde_json`, `jq` and
//! `JSON.parse`. Two files had drifted that way before this test existed:
//! `foreign-radiance` (one `Infinity`) and `foreign-uhdr` (six `NaN`).
//!
//! The repair quotes the token `json.dump` would have written bare. That is a
//! convention this file INTRODUCES, not one it inherits: nothing in the tree
//! had settled on one. `foreign-nifti` is the only other capture that records
//! a non-finite float, and it carries both spellings at once. Its `probe.c`
//! prints `"Infinity"` and `"NaN"` for the header floats it cannot spell as
//! numbers, while a `str(v)` in its `capture.py` writes `"inf"`, `"-inf"` and
//! `"nan"` for the `on_disk_pixdim` rows... same kind of data, different
//! token. Bringing it onto one spelling means re-capturing it, which is the
//! repin question #650 / #673 own rather than this one.
//!
//! So the encoding a reader has to agree with is pinned here rather than left
//! to each consumer:
//!
//!   * a finite value stays a JSON number,
//!   * the other three are the strings `"NaN"`, `"Infinity"` and
//!     `"-Infinity"`.
//!
//! `null` was the other candidate and is not usable: it collapses NaN, `+inf`
//! and `-inf` onto one value, and the whole point of the two records that
//! carry them is WHICH one libvips produced.

use std::fs;
use std::path::{Path, PathBuf};

/// Repo root (the directory containing the root `Cargo.toml`).
fn repo_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Every `oracle.json` under `oracle-captures/`, sorted. The tree is two
/// levels deep in one place (`convolution/canny`), so this recurses rather
/// than reading one directory.
fn oracle_files() -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(dir).expect("oracle-captures must be readable") {
            let path = entry.expect("readable directory entry").path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.file_name().is_some_and(|n| n == "oracle.json") {
                out.push(path);
            }
        }
    }

    let mut out = Vec::new();
    walk(&repo_root().join("oracle-captures"), &mut out);
    out.sort();

    // Named rather than counted. A floor of `>= 13` cannot fire while the tree
    // is pruned down to 13, and pinning the exact number turns every capture
    // anyone adds into an unrelated red test here. The list catches the two
    // things that would actually make the guard vacuous, a walk that quietly
    // finds nothing and an area that quietly disappears, and stays silent when
    // an area is added.
    let found: Vec<String> = out
        .iter()
        .map(|p| {
            p.strip_prefix(repo_root())
                .unwrap_or(p)
                .to_string_lossy()
                .replace('\\', "/")
        })
        .collect();
    for area in KNOWN_CAPTURES {
        assert!(
            found.iter().any(|f| f == area),
            "{area} is missing, so the walk is not seeing the whole tree. Found: {found:?}"
        );
    }
    out
}

/// Every capture area checked in today. See `oracle_files` for why this is a
/// list and not a count.
const KNOWN_CAPTURES: [&str; 14] = [
    "oracle-captures/convolution/canny/oracle.json",
    "oracle-captures/convolution/oracle.json",
    "oracle-captures/foreign-analyze/oracle.json",
    "oracle-captures/foreign-avif/oracle.json",
    "oracle-captures/foreign-exr/oracle.json",
    "oracle-captures/foreign-fits/oracle.json",
    "oracle-captures/foreign-gif/oracle.json",
    "oracle-captures/foreign-jp2k/oracle.json",
    "oracle-captures/foreign-jxl/oracle.json",
    "oracle-captures/foreign-mat/oracle.json",
    "oracle-captures/foreign-nifti/oracle.json",
    "oracle-captures/foreign-radiance/oracle.json",
    "oracle-captures/foreign-uhdr/oracle.json",
    "oracle-captures/foreign-webp/oracle.json",
];

/// Decode one recorded float: a JSON number, or one of the three quoted
/// tokens a capture writes for a value RFC 8259 has no literal for.
///
/// `at` names the file and the key, because the tree is not uniform yet.
/// `foreign-nifti` spells the same three values `"nan"`, `"inf"` and `"-inf"`
/// in its `on_disk_pixdim` rows, so pointing this at that capture is a
/// plausible next move and it should come back with a diagnosis rather than a
/// bare panic naming a token and no row.
fn recorded_float(v: &serde_json::Value, at: &str) -> f64 {
    match v {
        serde_json::Value::Number(n) => n
            .as_f64()
            .unwrap_or_else(|| panic!("{at}: the recorded float {n} does not fit an f64")),
        serde_json::Value::String(s) => match s.as_str() {
            "NaN" => f64::NAN,
            "Infinity" => f64::INFINITY,
            "-Infinity" => f64::NEG_INFINITY,
            other => panic!(
                "{at}: {other:?} is not one of the three recorded non-finite tokens \
                 (\"NaN\", \"Infinity\", \"-Infinity\"). foreign-nifti writes \
                 \"nan\", \"inf\" and \"-inf\" for the same values, so if that is \
                 the capture being read it has to be brought onto this spelling \
                 first (issue #674)"
            ),
        },
        other => panic!("{at}: {other} is not a recorded float"),
    }
}

/// The write half of the same convention, mirroring what `capture.py` does on
/// the way out.
fn record_float(v: f64) -> serde_json::Value {
    if v.is_finite() {
        return serde_json::json!(v);
    }
    if v.is_nan() {
        return serde_json::json!("NaN");
    }
    serde_json::json!(if v > 0.0 { "Infinity" } else { "-Infinity" })
}

/// The guard the two bad files walked past for as long as they existed: parse
/// every capture with a parser that implements the standard and nothing else.
///
/// It collects rather than failing on the first file, because the interesting
/// answer when this goes red is how far the rot spread, not which name sorts
/// first.
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn every_oracle_capture_is_strict_json() {
    let mut bad = Vec::new();
    for path in oracle_files() {
        let text = fs::read_to_string(&path).expect("a capture must be readable");
        if let Err(e) = serde_json::from_str::<serde_json::Value>(&text) {
            let rel = path.strip_prefix(repo_root()).unwrap_or(&path);
            bad.push(format!("{}: {e}", rel.display()));
        }
    }
    assert!(
        bad.is_empty(),
        "these captures are not JSON any standard parser accepts. \
         A non-finite float is written as \"NaN\", \"Infinity\" or \
         \"-Infinity\", never bare (issue #674):\n  {}",
        bad.join("\n  ")
    );
}

/// The parser the test above leans on has to actually be strict, or that test
/// is a check that cannot fail for the reason it claims to. `serde_json`
/// rejects the three bare literals; this says so out loud so a future switch
/// to a lenient parser shows up here rather than as silence.
#[test]
fn the_strict_parser_really_does_reject_bare_literals() {
    for bare in ["[NaN]", "[Infinity]", "[-Infinity]"] {
        assert!(
            serde_json::from_str::<serde_json::Value>(bare).is_err(),
            "{bare} parsed, so the strict-JSON guard is testing nothing"
        );
    }
}

/// The three tokens are pairwise distinct and survive write-then-read with the
/// value's identity intact, which is exactly what `null` would not give: NaN,
/// `+inf` and `-inf` come back as themselves and not as each other.
///
/// Worth naming what this does NOT prove, because the name it used to have
/// claimed more. Both halves live in this file and `record_float` has no other
/// caller anywhere in the tree, so this is two functions written together
/// agreeing with each other. It says nothing about whether `capture.py`'s
/// `json_safe` writes the same three tokens... that is
/// `the_repaired_captures_still_record_a_real_infinity_and_a_real_nan` below,
/// which reads what the encoder actually wrote to disk.
#[test]
fn the_three_recorded_tokens_are_distinct_and_survive_a_round_trip() {
    let values = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        -1.5,
        f64::MAX,
    ];

    let encoded: Vec<serde_json::Value> = values.iter().copied().map(record_float).collect();
    let text = serde_json::to_string(&encoded).expect("the encoding must serialise");
    assert!(
        !text.contains("NaN,") && !text.contains("[NaN") && !text.contains("Infinity,"),
        "the encoded form still carries a bare literal: {text}"
    );

    let parsed: Vec<serde_json::Value> =
        serde_json::from_str(&text).expect("the encoded form must be strict JSON");
    let read_back: Vec<f64> = parsed
        .iter()
        .enumerate()
        .map(|(i, v)| recorded_float(v, &format!("this test's own encoded[{i}]")))
        .collect();

    assert_eq!(read_back.len(), values.len());
    for (before, after) in values.iter().zip(&read_back) {
        if before.is_nan() {
            assert!(after.is_nan(), "NaN came back as {after}");
        } else {
            assert_eq!(
                before.to_bits(),
                after.to_bits(),
                "{before} came back as {after}"
            );
        }
    }

    // The three tokens are pairwise distinct on the wire, so nothing about the
    // encoding is lossy: a reader can tell which one it has.
    let tokens: Vec<&serde_json::Value> = encoded[..3].iter().collect();
    assert_ne!(tokens[0], tokens[1]);
    assert_ne!(tokens[1], tokens[2]);
    assert_ne!(tokens[0], tokens[2]);
}

/// The two records the repair touched still say what they measured. Quoting
/// the token must not have quietly turned an infinity into a string nobody
/// decodes, or into the `0.0` that sits next to it in the same row.
#[test]
#[cfg_attr(miri, ignore)] // filesystem access blocked by Miri isolation
fn the_repaired_captures_still_record_a_real_infinity_and_a_real_nan() {
    let radiance: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(repo_root().join("oracle-captures/foreign-radiance/oracle.json"))
            .expect("the radiance capture must be readable"),
    )
    .expect("the radiance capture must be strict JSON");

    // `float2rad`'s setcolr sweep: row 8 is the undefined-behaviour row, and
    // its first component is the +inf that makes it one.
    let row = &radiance["records"]["encode_setcolr"]["inputs"][8];
    let at =
        |band| format!("foreign-radiance/oracle.json records.encode_setcolr.inputs[8][{band}]");
    let inf = recorded_float(&row[0], &at(0));
    assert!(
        inf.is_infinite() && inf.is_sign_positive(),
        "encode_setcolr input 8 must be +inf, got {inf}"
    );
    assert_eq!(recorded_float(&row[1], &at(1)), 1.0);
    assert_eq!(recorded_float(&row[2], &at(2)), 1.0);

    let uhdr: serde_json::Value = serde_json::from_str(
        &fs::read_to_string(repo_root().join("oracle-captures/foreign-uhdr/oracle.json"))
            .expect("the uhdr capture must be readable"),
    )
    .expect("the uhdr capture must be strict JSON");

    // `log2(min) * (1.0f - gg)` is `-inf * 0` at both ends of the degenerate
    // metadata sweep, so one pixel per arm is NaN in all three bands. Which
    // pixel differs: the min arm lands on the last, the max arm on the first.
    let results = &uhdr["records"]["uhdr2scRGB_degenerate_metadata"]["results"];
    for (arm, pixel) in [("min_boost_zero", 15), ("max_boost_zero", 0)] {
        let scrgb = &results[arm]["scRGB"];
        assert_eq!(scrgb.as_array().expect("a 16-pixel row").len(), 16);
        for band in 0..3 {
            let v = recorded_float(
                &scrgb[pixel][band],
                &format!(
                    "foreign-uhdr/oracle.json records.uhdr2scRGB_degenerate_metadata\
                     .results.{arm}.scRGB[{pixel}][{band}]"
                ),
            );
            assert!(
                v.is_nan(),
                "{arm} pixel {pixel} band {band} must be NaN, got {v}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The other half of the same problem (issue #682).
//
// Everything above reads what a capture WROTE. It catches a bare `NaN` the
// moment CI runs, which is a real net and is what found the two bad files.
// What it cannot do is stop the capture producing one, and by the time CI is
// red the person who ran `python3 capture.py` has moved on and the machine
// with libvips on it is somebody else's laptop.
//
// `json.dump(..., allow_nan=False)` moves that failure to the write, where it
// costs a re-run and no investigation. Two of the fourteen capture scripts set
// it, both from #674. The rest are guarded here.
// ---------------------------------------------------------------------------

/// Every Python file under `oracle-captures/`, sorted.
///
/// All of them, not just `capture.py`. `oracle_pin.py` serialises nothing
/// today, and a helper that starts writing JSON tomorrow should not have to be
/// added to a list before this notices.
fn capture_scripts() -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(dir).expect("oracle-captures must be readable") {
            let path = entry.expect("readable directory entry").path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|e| e == "py") {
                out.push(path);
            }
        }
    }

    let mut out = Vec::new();
    walk(&repo_root().join("oracle-captures"), &mut out);
    out.sort();
    out
}

/// Blank out Python comments and the INSIDES of string literals, keeping every
/// other byte and every newline where it was.
///
/// This is what stops the scan below being the kind of check this repository
/// keeps having to unpick. A `grep allow_nan` passes on a file whose only
/// mention of it is the comment explaining why it matters, and both scripts
/// that already set the flag carry exactly such a comment. Blanking first
/// means prose cannot answer for code: what survives is the syntax.
///
/// The quote characters themselves stay, so a call's argument list still reads
/// as a list of values. Backslash escapes are honoured, which is right for raw
/// strings too: `r"\""` does not end at the second quote in Python either.
///
/// An f-string is two things at once and is handled as both. Its literal text
/// is prose and gets blanked, and what sits inside its `{...}` fields is code
/// and is kept, because `foreign-jp2k` writes one of its four dict-key dumps
/// there:
///
/// ```text
/// body = ",\n".join(f"{pad}  {json.dumps(k)}: {encode(v, indent + 2)}"
/// ```
///
/// Blanking the whole literal hides that call, and the first version of this
/// scan did exactly that: it reported seventeen unguarded call sites where the
/// tree has eighteen, and the one it lost was the one hiding inside a string.
fn code_only(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = vec![b' '; b.len()];
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        if c == b'\n' {
            out[i] = b'\n';
            i += 1;
        } else if c == b'#' {
            while i < b.len() && b[i] != b'\n' {
                i += 1;
            }
        } else if c == b'"' || c == b'\'' {
            // A string prefix is at most two of `rRbBuUfF` and sits directly
            // against the quote, so scanning back over identifier bytes tells
            // an f-string from an `elif` that happens to end in an f.
            let mut p = i;
            while p > 0 && (b[p - 1].is_ascii_alphanumeric() || b[p - 1] == b'_') {
                p -= 1;
            }
            let prefix = &b[p..i];
            let is_f = prefix.len() <= 2
                && prefix.iter().all(|c| b"rRbBuUfF".contains(c))
                && prefix.iter().any(|c| *c == b'f' || *c == b'F');

            let triple = i + 2 < b.len() && b[i + 1] == c && b[i + 2] == c;
            let quote = if triple { 3 } else { 1 };
            out[i..(i + quote).min(b.len())].fill(c);
            let mut j = i + quote;
            let mut field = 0usize;
            let close = loop {
                if j >= b.len() {
                    break None;
                }
                if b[j] == b'\\' {
                    if b.get(j + 1) == Some(&b'\n') {
                        out[j + 1] = b'\n';
                    }
                    j += 2;
                    continue;
                }
                if is_f && field == 0 && (b[j] == b'{' || b[j] == b'}') {
                    // `{{` and `}}` are literal braces, not a field.
                    if b.get(j + 1) == Some(&b[j]) {
                        j += 2;
                        continue;
                    }
                    if b[j] == b'{' {
                        field = 1;
                    }
                    j += 1;
                    continue;
                }
                if is_f && field > 0 {
                    match b[j] {
                        b'{' => field += 1,
                        b'}' => field -= 1,
                        b'\n' => out[j] = b'\n',
                        other => out[j] = other,
                    }
                    j += 1;
                    continue;
                }
                if b[j] == c && (!triple || (b.get(j + 1) == Some(&c) && b.get(j + 2) == Some(&c)))
                {
                    break Some(j);
                }
                if b[j] == b'\n' {
                    out[j] = b'\n';
                }
                j += 1;
            };
            match close {
                Some(j) => {
                    out[j..(j + quote).min(b.len())].fill(c);
                    i = (j + quote).min(b.len());
                }
                // An unterminated literal is a syntax error the capture would
                // have hit long before this test ran, so there is nothing
                // sensible left to scan.
                None => i = b.len(),
            }
        } else {
            out[i] = c;
            i += 1;
        }
    }
    String::from_utf8(out).expect("blanking only ever replaces whole bytes with ASCII")
}

/// One `json.dump` / `json.dumps` call: its 1-based line, and the text of its
/// own argument list.
struct DumpCall {
    line: usize,
    args: String,
}

/// Find every `json.dump(` and `json.dumps(` in already-blanked source, and
/// take each call's arguments by matching its own brackets.
///
/// Per call rather than per file, because `foreign-avif` and `foreign-jp2k`
/// have four apiece: both hand-roll an encoder that keeps a leaf array on one
/// line, and a leaf is exactly where a float lives. A file-wide answer would
/// call those two guarded while three quarters of their serialisation was not.
///
/// Nesting is half resolved here. A `json.dumps` inside another call's
/// arguments is its own site with its own brackets, so it cannot borrow the
/// outer call's flag. The opposite direction is NOT handled here and must not
/// be read into this: the outer call's `args` still contain the inner call
/// text in full, flag and all, which is why [`refuses_non_finite`] searches
/// only at bracket depth 0.
fn json_dump_calls(code: &str) -> Vec<DumpCall> {
    let b = code.as_bytes();
    let mut out = Vec::new();
    let mut from = 0;
    while let Some(hit) = code[from..].find("json.dump") {
        let start = from + hit;
        let after = start + "json.dump".len();
        from = after;
        let open = if code[after..].starts_with('(') {
            after
        } else if code[after..].starts_with("s(") {
            after + 1
        } else {
            continue;
        };

        let mut depth = 0usize;
        let mut k = open;
        let close = loop {
            if k >= b.len() {
                break None;
            }
            match b[k] {
                b'(' | b'[' | b'{' => depth += 1,
                b')' | b']' | b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        break Some(k);
                    }
                }
                _ => {}
            }
            k += 1;
        };
        let close = close.unwrap_or_else(|| {
            panic!("a json.dump call starting at byte {start} never closes its brackets")
        });

        out.push(DumpCall {
            line: 1 + code[..start].bytes().filter(|c| *c == b'\n').count(),
            args: code[open + 1..close].to_string(),
        });
        // `from` stays just past this call's name rather than past its closing
        // bracket, so a `json.dumps` nested in another call's arguments is
        // still found and still answered on its own brackets.
    }
    out
}

/// Does this argument list pass `allow_nan=False` as an argument of ITS OWN
/// call?
///
/// Whitespace is squashed so `allow_nan = False` counts, and the argument text
/// has already had its string interiors emptied, so nothing a capture happens
/// to SAY can answer for what it does.
///
/// The depth filter is the part that stops this being a false pass. `args` is
/// the whole text between the call's own brackets, and a nested call's
/// arguments are inside it, so
///
/// ```python
/// json.dump(o, f, default=lambda v: json.dumps(v, allow_nan=False))
/// ```
///
/// carries the string while the outer `json.dump` is bare, and a bare NaN
/// walks straight out of it. A keyword argument of this call always sits at
/// depth 0 of this call's argument list, so dropping every bracketed group
/// keeps every real spelling and loses exactly the borrowed ones.
///
/// This is the direction that matters, because it is the one that produces a
/// PASS. The other direction (an inner call reading the outer call's flag)
/// cannot happen at all: [`json_dump_calls`] gives the inner call its own
/// brackets, so its `args` never contains the outer call's text.
///
/// One contrived spelling still reads as guarded and is left that way on
/// purpose: `allow_nan=False if strict else True` contains the substring at
/// depth 0. Nothing in the tree writes anything like it, matching the
/// expression rather than the substring means parsing Python here, and every
/// other way of getting the flag wrong (`allow_nan=True`, a keyword built at
/// runtime, the flag omitted) fails in the loud direction and is caught.
fn refuses_non_finite(args: &str) -> bool {
    let mut depth = 0usize;
    let mut top = String::with_capacity(args.len());
    for c in args.chars() {
        match c {
            '(' | '[' | '{' => depth += 1,
            // `saturating_sub` because a call whose arguments open with a
            // closing bracket is not something this scanner should panic on:
            // it cannot arise from `json_dump_calls`, which matches brackets,
            // but a future caller handing this a fragment should get a "no"
            // rather than a crash.
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            _ if depth == 0 && !c.is_whitespace() => top.push(c),
            _ => {}
        }
    }
    top.contains("allow_nan=False")
}

/// How many `json.dump` / `json.dumps` call sites the capture scripts hold,
/// across all of them.
///
/// The per-script floor below only catches a scanner that finds NOTHING in a
/// file. It cannot catch one that loses some sites, and that is not a
/// hypothetical: the first version of `code_only` blanked f-strings whole and
/// reported 17 unguarded call sites where the tree had 18. Every per-script
/// floor was green through that, because every script still yielded at least
/// one. The count is what caught it, so the count belongs in the test.
///
/// Re-derive it with Python's own parser rather than by counting the greps,
/// which over-count: 27 lines under `oracle-captures/` mention `json.dump` and
/// 7 of those are prose in a docstring or a comment.
///
/// ```text
/// python3 - <<'PY'
/// import ast, pathlib
/// n = 0
/// for path in sorted(pathlib.Path("oracle-captures").rglob("*.py")):
///     for node in ast.walk(ast.parse(path.read_text())):
///         if (isinstance(node, ast.Call)
///                 and isinstance(node.func, ast.Attribute)
///                 and node.func.attr in ("dump", "dumps")
///                 and isinstance(node.func.value, ast.Name)
///                 and node.func.value.id == "json"):
///             n += 1
/// print(n)
/// PY
/// ```
///
/// That printed 20 on CPython 3.14.7, which is where this number comes from,
/// and the Rust scanner agrees with it site for site. Two scripts hold four
/// apiece (`foreign-avif` and `foreign-jp2k`, both hand-rolling an encoder),
/// twelve hold one, and `oracle_pin.py` holds none.
///
/// When a capture script is legitimately added, run that walk again and put
/// the new number here in the same commit that adds the script. Do not turn
/// this into a `>=`: a floor cannot fire while the tree shrinks past it, which
/// is the exact failure it is here to catch.
const EXPECTED_DUMP_CALL_SITES: usize = 20;

/// Every `json.dump` / `json.dumps` call site in the capture scripts passes
/// `allow_nan=False`, so the write stops rather than emitting a bare
/// non-finite float (issue #682).
///
/// The rule is blanket: every call in the tree passes the flag, including the
/// two that only serialise a dict key and cannot go non-finite. An exemption
/// needs a rule for who qualifies, and any such rule is a thing you can argue
/// your way past six months later. This one has nothing to argue with.
///
/// # What the name deliberately does not claim
///
/// It says *call site*, not *capture script*, because `allow_nan` only reaches
/// a float that json serialises. A script that turns a float into text itself
/// and hands json a string walks past this, and the flag was never going to
/// stop it.
///
/// I measured how far that reaches rather than guessing, by walking every
/// committed `oracle.json` for string leaves that spell a float, and then
/// tracing each one back to the line that wrote it:
///
/// * **Non-finite by a route this cannot see: `foreign-nifti` alone, two
///   leaves.** `capture.py:607` writes `str(v)` for a pixdim the header holds
///   as `inf` or `nan`, which lands as lowercase `"inf"` / `"nan"` rather than
///   the `"Infinity"` / `"NaN"` this file pins. That is the mismatch the
///   module doc opens with, and bringing it onto one spelling means
///   re-capturing the area, which is #650 / #673's question and not this one's.
///   Its other fifteen non-finite string leaves are not this shape at all:
///   four are `probe.c`'s `jnum` printing `Infinity` and `NaN` for a header
///   float, and eleven are hand-written labels (`FLOAT_VALUES_LABELS` and the
///   `values_packed` rows) spelling out the input bytes.
/// * **Non-finite through the convention helper, which is the point rather
///   than a gap:** `foreign-radiance` (one) and `foreign-uhdr` (six) go through
///   their `json_safe`, which is #674's repair and produces exactly the pinned
///   spelling.
/// * **Finite floats turned into text: `foreign-uhdr` alone**, at
///   `capture.py:672`, `repr(lut[i])` over the 256-entry `vips_v2Y_8` sRGB
///   table so the decimals round-trip exactly. That table is
///   `f/12.92` or `powf((f+0.055)/1.055, 2.4)` over `f` in `[0, 1]`, so it
///   cannot go non-finite. Every other float-looking string in the tree is
///   text something else produced: a libvips header value, a `vips getpoint`
///   stdout, a brew version, a sigma spelled on a command line.
///
/// So one script writes a non-finite float outside a `json.dump` today, for a
/// reason already written down and owned elsewhere. Widening the scan to catch
/// it would mean deciding statically whether an arbitrary `str()`, `repr()`,
/// f-string or `%` format receives a float and whether its result reaches the
/// oracle. There are 394 of those constructs across the 15 scripts, so any
/// such rule either fires on all of them or misses the next spelling, and it
/// would be a worse guard than an honest name.
#[test]
#[cfg_attr(miri, ignore)] // reads the capture scripts on disk, which Miri isolation blocks
fn every_json_dump_call_site_refuses_a_non_finite_float() {
    let scripts = capture_scripts();
    let mut sites_by_script: Vec<(String, usize)> = Vec::new();
    let mut unguarded: Vec<String> = Vec::new();

    for path in &scripts {
        let rel = path
            .strip_prefix(repo_root())
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        let src = fs::read_to_string(path).expect("a capture script must be readable");
        let calls = json_dump_calls(&code_only(&src));
        sites_by_script.push((rel.clone(), calls.len()));
        for call in &calls {
            if !refuses_non_finite(&call.args) {
                unguarded.push(format!("{rel}:{}", call.line));
            }
        }
    }

    // The scan is the part that can quietly find nothing and take the whole
    // test with it, so it is anchored before any conclusion is drawn. Every
    // capture area has a `capture.py` next to its `oracle.json`, and every one
    // of those writes JSON, so a script that yields no call site means the
    // scanner broke rather than that the script is clean.
    for capture in KNOWN_CAPTURES {
        let script = capture.replace("oracle.json", "capture.py");
        let found = sites_by_script
            .iter()
            .find(|(rel, _)| *rel == script)
            .unwrap_or_else(|| panic!("{script} was not scanned; found {sites_by_script:?}"));
        assert!(
            found.1 > 0,
            "{script} yielded no json.dump call site. Every capture writes its \
             oracle.json with one, so this is the scanner failing to see the \
             code rather than the script having none."
        );
    }

    // ...and the total, which is the column the per-script floor cannot cover.
    // A scanner that stops seeing SOME sites keeps every script above zero and
    // walks past the loop above; it cannot walk past this. See
    // `EXPECTED_DUMP_CALL_SITES` for how to re-derive the number.
    let total: usize = sites_by_script.iter().map(|(_, n)| n).sum();
    assert_eq!(
        total, EXPECTED_DUMP_CALL_SITES,
        "the scan found {total} json.dump/json.dumps call sites across the \
         capture scripts, not {EXPECTED_DUMP_CALL_SITES}. If a capture script \
         was added or removed, re-derive the number with the `ast` walk in the \
         doc on `EXPECTED_DUMP_CALL_SITES` and update it in the same commit. \
         If no script changed, the scanner has stopped seeing part of the \
         tree, which is what this pin is for. Per script: {sites_by_script:?}"
    );

    assert!(
        unguarded.is_empty(),
        "these json.dump/json.dumps calls can write a bare NaN, Infinity or \
         -Infinity, none of which is JSON, and Python will read the result \
         back without complaining so the damage surfaces in another language \
         months later (issue #682). Pass allow_nan=False so the capture stops \
         at the write instead:\n  {}",
        unguarded.join("\n  ")
    );
}

/// The scan really does read code and not prose, and really does answer per
/// call.
///
/// Without this, `every_json_dump_call_site_refuses_a_non_finite_float`
/// is a check whose fallibility nobody has looked at: a scanner that returned
/// "guarded" for everything would pass it just as well as a correct one, and
/// the two scripts that already set the flag both carry a COMMENT saying
/// `allow_nan=False`, which is precisely the string a naive grep would find.
#[test]
fn the_allow_nan_scan_cannot_be_satisfied_by_a_comment_or_a_docstring() {
    let decoy = concat!(
        "import json\n",
        "\n",
        "def write(oracle, f):\n",
        "    \"\"\"Dumps with allow_nan=False so a non-finite stops us.\"\"\"\n",
        "    # allow_nan=False\n",
        "    json.dump(oracle, f, indent=2)\n",
    );
    let calls = json_dump_calls(&code_only(decoy));
    assert_eq!(calls.len(), 1, "expected one call site in the decoy");
    assert_eq!(calls[0].line, 6, "the reported line must be the call's");
    assert!(
        !refuses_non_finite(&calls[0].args),
        "a docstring and a comment satisfied the scan, so it is checking prose"
    );

    // The real form, and the spaced spelling of it.
    for guarded in [
        "json.dump(oracle, f, indent=2, allow_nan=False)\n",
        "json.dump(oracle, f, allow_nan = False)\n",
    ] {
        let calls = json_dump_calls(&code_only(guarded));
        assert_eq!(calls.len(), 1);
        assert!(refuses_non_finite(&calls[0].args), "{guarded} should pass");
    }

    // `allow_nan=True` is the flag written out and turned off, which is worse
    // than not writing it, and must not read as guarded.
    let calls = json_dump_calls(&code_only("json.dump(o, f, allow_nan=True)\n"));
    assert_eq!(calls.len(), 1);
    assert!(!refuses_non_finite(&calls[0].args));

    // Half a file guarded is not a guarded file: the encoders in foreign-avif
    // and foreign-jp2k have four call sites each.
    let half = "json.dumps(a, allow_nan=False)\njson.dumps(b, indent=2)\n";
    let calls = json_dump_calls(&code_only(half));
    assert_eq!(calls.len(), 2);
    assert!(refuses_non_finite(&calls[0].args));
    assert!(
        !refuses_non_finite(&calls[1].args),
        "the second call is bare"
    );

    // A nested call cannot borrow the outer call's flag.
    let nested = "json.dumps(json.dumps(x), allow_nan=False)\n";
    let calls = json_dump_calls(&code_only(nested));
    assert_eq!(calls.len(), 2, "the inner call is its own site");
    assert!(
        refuses_non_finite(&calls[0].args),
        "the outer call is guarded"
    );
    assert!(!refuses_non_finite(&calls[1].args), "the inner call is not");

    // ...and the direction that produces a false PASS, which is the one that
    // matters: the INNER call carries the flag and the outer one does not.
    // `args` is the whole text between the outer call's own brackets, nested
    // calls included, so a plain substring search reads the outer call as
    // guarded and a real bare NaN walks straight out of it.
    let borrowed = "json.dump(o, f, default=lambda v: json.dumps(v, allow_nan=False))\n";
    let calls = json_dump_calls(&code_only(borrowed));
    assert_eq!(calls.len(), 2, "the inner call is its own site");
    assert!(
        !refuses_non_finite(&calls[0].args),
        "the outer call borrowed the flag from the call nested in its arguments"
    );
    assert!(
        refuses_non_finite(&calls[1].args),
        "the inner call is the one that is guarded"
    );

    // A string mentioning json.dump is not a call site.
    let quoted = "note = \"json.dump(o, f) writes a bare NaN\"\n";
    assert!(
        json_dump_calls(&code_only(quoted)).is_empty(),
        "a call named inside a string counted as one"
    );

    // ...but a call inside an f-string field IS one, which is where
    // foreign-jp2k keeps its fourth. The literal text around it is still prose:
    // the `allow_nan=False` in it must not answer for the call.
    let fstring = "body = f\"{pad} allow_nan=False {json.dumps(k)}: x\"\n";
    let calls = json_dump_calls(&code_only(fstring));
    assert_eq!(calls.len(), 1, "a call in an f-string field is a call site");
    assert!(
        !refuses_non_finite(&calls[0].args),
        "the f-string's own text answered for the call inside it"
    );
}
