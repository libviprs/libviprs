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
    let _ = source;
    (0, Vec::new())
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

    // 23 at the time of writing. A floor well under that, so the gate cannot pass by scanning
    // nothing, without breaking every time a helper is added or folded away.
    assert!(
        seen >= 15,
        "the scan only found {seen} pdfium-render functions"
    );
    assert!(
        unguarded.is_empty(),
        "these use pdfium-render without taking pdfium_lock() first or documenting that the \
         caller holds it: {unguarded:?}"
    );
}
