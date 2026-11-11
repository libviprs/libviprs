//! Every function in `src/pdf.rs` and `src/streaming.rs` that uses pdfium-render must take
//! `pdfium_lock()` before its first use, or say in its doc comment that its caller holds it.
//!
//! The `pdfium-render` dependency comment in `Cargo.toml` relies on this, because the 0.9.4
//! bindings do not lock every method. This is a text scan, not a parser: it is crude on purpose
//! and should fail loudly on a false positive rather than miss a real gap. It reads source only,
//! so it needs no libpdfium and no feature flags.

use std::fs;
use std::path::Path;

/// Returns how many functions in `source` use pdfium-render, and the names of the ones that
/// neither take `pdfium_lock()` before their first use nor document that the caller holds it.
/// Anything after the `#[cfg(test)]` module marker is ignored.
fn scan(source: &str) -> (usize, Vec<String>) {
    let source = match source.find("#[cfg(test)]\nmod ") {
        Some(end) => &source[..end],
        None => source,
    };
    // Raw pdfium-render calls, any pdfium-render type in the signature or body, and calls to the
    // crate's own helpers whose docs say the caller holds the lock. The last group is what
    // catches a new function that reaches pdfium only through those helpers.
    let markers = [
        ".load_pdf_from",
        ".pages()",
        ".objects()",
        ".render_with_config(",
        ".as_raw_bytes(",
        "Pdfium::new(",
        "pdfium_render::",
        "&Pdfium",
        "&PdfPage",
        "&PdfDocument",
        "&PdfBitmap",
        "render_at_size(",
        "render_at_size_with_background(",
        "load_pdfium_document(",
        "render_page_strip_with_page(",
        "probe_page_dims(",
    ];

    let lines: Vec<&str> = source.lines().collect();
    let mut seen = 0;
    let mut unguarded = Vec::new();
    let mut doc = String::new();

    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();

        if let Some(text) = trimmed.strip_prefix("///") {
            doc.push_str(text);
            doc.push(' ');
            continue;
        }

        if trimmed.starts_with("#[") {
            continue;
        }

        let mut rest = trimmed;
        for prefix in [
            "pub(crate) ",
            "pub(super) ",
            "pub ",
            "const ",
            "async ",
            "unsafe ",
        ] {
            rest = rest.strip_prefix(prefix).unwrap_or(rest);
        }

        let Some(after_fn) = rest.strip_prefix("fn ") else {
            doc.clear();
            continue;
        };
        let name: String = after_fn
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();

        // Take the text from this line on, up to the brace that closes the body.
        let text = lines[index..].join("\n");
        let Some(open) = text
            .find(['{', ';'])
            .filter(|at| text[*at..].starts_with('{'))
        else {
            doc.clear();
            continue;
        };
        let mut depth = 0;
        let mut end = text.len();
        let bytes = text.as_bytes();
        let mut at = open;
        while at < bytes.len() {
            match bytes[at] {
                b'/' if bytes.get(at + 1) == Some(&b'/') => {
                    while at < bytes.len() && bytes[at] != b'\n' {
                        at += 1;
                    }
                    continue;
                }
                b'"' => {
                    at += 1;
                    while at < bytes.len() && bytes[at] != b'"' {
                        at += if bytes[at] == b'\\' { 2 } else { 1 };
                    }
                }
                b'\'' if bytes.get(at + 2) == Some(&b'\'') => at += 2,
                b'\'' if bytes.get(at + 1) == Some(&b'\\') && bytes.get(at + 3) == Some(&b'\'') => {
                    at += 3
                }
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = at;
                        break;
                    }
                }
                _ => {}
            }
            at += 1;
        }

        // The signature counts as well as the body, so a `&PdfPage` parameter is a use.
        // Uses in the signature (a `&PdfPage` parameter) count as a use that needs the caller's
        // lock. Uses in the body count from the first non-`use` line, so a lock taken at the top
        // of a function that also names a pdfium-render type in its return type still passes.
        let signature = &text[..open];
        let body: String = text[open..end]
            .lines()
            .map(|l| l.split("//").next().unwrap_or(""))
            .filter(|l| !l.trim_start().starts_with("use "))
            .collect::<Vec<_>>()
            .join("\n");
        let signature_use = markers.iter().any(|m| signature.contains(m));
        let first_use = markers.iter().filter_map(|m| body.find(m)).min();

        if signature_use || first_use.is_some() {
            seen += 1;
            let lock_at = ["pdfium_lock()", "&INIT_GUARD"]
                .iter()
                .filter_map(|lock| body.find(lock))
                .min();
            let locks_first = lock_at.is_some_and(|at| first_use.is_none_or(|use_at| at < use_at));
            let doc_text = doc.split_whitespace().collect::<Vec<_>>().join(" ");
            let caller_holds =
                doc_text.contains("caller holds") && doc_text.contains("pdfium_lock");
            if !locks_first && !caller_holds {
                unguarded.push(name);
            }
        }
        doc.clear();
    }

    (seen, unguarded)
}

#[test]
fn flags_a_function_that_renders_without_the_lock() {
    let source = r#"
fn render_unlocked(path: &std::path::Path) {
    let pdfium = init_pdfium().unwrap();
    let document = pdfium.load_pdf_from_file(path, None).unwrap();
    let _ = document.pages().len();
}
"#;

    let (seen, unguarded) = scan(source);

    assert_eq!(seen, 1);
    assert_eq!(unguarded, vec!["render_unlocked".to_string()]);
}

#[test]
fn flags_a_function_that_reaches_pdfium_only_through_a_caller_holds_helper() {
    let source = r#"
fn render_via_helper(page: &PdfPage<'_>) -> Raster {
    render_at_size(page, 100, 100).unwrap()
}

fn render_via_helper_without_a_handle_type() -> Raster {
    let page = something();
    render_at_size(page, 100, 100).unwrap()
}
"#;

    let (seen, unguarded) = scan(source);

    assert_eq!(seen, 2);
    assert_eq!(
        unguarded,
        vec![
            "render_via_helper".to_string(),
            "render_via_helper_without_a_handle_type".to_string()
        ]
    );
}

#[test]
fn flags_a_function_that_only_takes_a_pdfium_render_type() {
    let source = r#"
fn takes_a_document(doc: &pdfium_render::prelude::PdfDocument<'_>) -> usize {
    0
}

fn locks_but_names_a_type_in_its_return(path: &std::path::Path) -> pdfium_render::prelude::PdfColor {
    use pdfium_render::prelude::*;
    let _lock = pdfium_lock();
    PdfColor::WHITE
}
"#;

    let (seen, unguarded) = scan(source);

    assert_eq!(seen, 2);
    assert_eq!(unguarded, vec!["takes_a_document".to_string()]);
}

#[test]
fn accepts_a_function_that_locks_before_calling_a_caller_holds_helper() {
    let source = r#"
fn render_locked_via_helper(path: &std::path::Path) -> Raster {
    let _lock = pdfium_lock();
    let page = open_page(path);
    render_at_size(page, 100, 100).unwrap()
}
"#;

    let (seen, unguarded) = scan(source);

    assert_eq!(seen, 1);
    assert!(unguarded.is_empty(), "{unguarded:?}");
}

#[test]
fn accepts_a_function_that_takes_the_lock_first() {
    let source = r#"
fn render_locked(path: &std::path::Path) {
    let _lock = pdfium_lock();
    let pdfium = init_pdfium().unwrap();
    let document = pdfium.load_pdf_from_file(path, None).unwrap();
    let _ = document.pages().len();
}
"#;

    let (seen, unguarded) = scan(source);

    assert_eq!(seen, 1);
    assert!(unguarded.is_empty(), "{unguarded:?}");
}

#[test]
fn flags_a_function_that_takes_the_lock_after_its_first_use() {
    let source = r#"
fn render_late(path: &std::path::Path) {
    let pdfium = init_pdfium().unwrap();
    let document = pdfium.load_pdf_from_file(path, None).unwrap();
    let _lock = pdfium_lock();
    let _ = document.pages().len();
}
"#;

    let (_, unguarded) = scan(source);

    assert_eq!(unguarded, vec!["render_late".to_string()]);
}

#[test]
fn accepts_a_function_whose_doc_says_the_caller_holds_the_lock() {
    let source = r#"
/// Loads the document. The caller holds [`pdfium_lock`].
#[cfg(feature = "pdfium")]
fn load_helper<'a>(pdfium: &'a Pdfium, path: &std::path::Path) -> PdfDocument<'a> {
    pdfium.load_pdf_from_file(path, None).unwrap()
}
"#;

    let (seen, unguarded) = scan(source);

    assert_eq!(seen, 1);
    assert!(unguarded.is_empty(), "{unguarded:?}");
}

#[test]
fn ignores_functions_in_the_test_module() {
    let source = r#"
#[cfg(test)]
mod tests {
    fn render_unlocked_in_a_test(pdfium: &Pdfium) {
        let _ = pdfium.load_pdf_from_file("a.pdf", None);
    }
}
"#;

    assert_eq!(scan(source), (0, Vec::new()));
}

#[test]
#[cfg_attr(miri, ignore)] // reads src/ from disk, blocked by Miri isolation
fn every_pdfium_function_in_src_is_locked_or_documented() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut seen = 0;
    let mut unguarded = Vec::new();

    for file in ["src/pdf.rs", "src/streaming.rs"] {
        let source = fs::read_to_string(root.join(file)).unwrap();
        let (count, names) = scan(&source);
        seen += count;
        unguarded.extend(names.into_iter().map(|name| format!("{file}::{name}")));
    }

    // 25 at the time of writing (the 23 in #1602 plus two pure helpers that take a pdfium-render
    // type). A floor just under that, so the gate cannot pass by scanning only the functions that
    // call a raw marker, without breaking every time a helper is added or folded away.
    assert!(
        seen >= 23,
        "the scan only found {seen} pdfium-render functions"
    );
    assert!(
        unguarded.is_empty(),
        "these use pdfium-render without taking pdfium_lock() first or documenting that the \
         caller holds it: {unguarded:?}"
    );
}
