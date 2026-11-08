//! Issue #1578 (L1, the part that needs none of the new streams): `pvec::pdf_page_to_pvec` is the
//! mapping the server's `convert_pdf` has always carried, moved into the library. The bytes it
//! writes for `blueprint-mix.pdf` are pinned to what that mapping wrote before the move (taken
//! with `pvec-container` at the facade's pin and at the server's, which agree). Needs
//! `--features pvec-write`.

#![cfg(feature = "pvec-write")]

use libviprs::pvec::{PdfVectorLimits, extract_page_paths, pdf_page_to_pvec};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

const GOLDEN_LEN: usize = 14336;
const GOLDEN_SHA256: &str = "ed21b32641dd385c5170247766299f0a31c216e4d71ec944a2ec849299a6b802";

fn blueprint_mix() -> Option<PathBuf> {
    let fixtures = std::env::var_os("LIBVIPRS_TESTS_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../libviprs-tests/tests/fixtures")
        });
    let path = fixtures.join("blueprint-mix.pdf");
    if path.is_file() {
        Some(path)
    } else {
        eprintln!(
            "skipped: {} is not there (set LIBVIPRS_TESTS_FIXTURES)",
            path.display()
        );
        None
    }
}

#[test]
fn blueprint_mix_bytes_match_the_mapping_the_server_had() {
    let Some(path) = blueprint_mix() else { return };
    let page = extract_page_paths(&path, 1, &PdfVectorLimits::default()).unwrap();
    let bytes = pdf_page_to_pvec(&page, "iasbuilt-server", "0.0.0").unwrap();
    assert_eq!(bytes.len(), GOLDEN_LEN);
    assert_eq!(hex(&Sha256::digest(&bytes)), GOLDEN_SHA256);
}

#[test]
fn a_page_with_nothing_painted_is_refused_with_the_servers_message() {
    let Some(path) = blueprint_mix() else { return };
    let mut page = extract_page_paths(&path, 1, &PdfVectorLimits::default()).unwrap();
    page.entities.clear();
    let error = pdf_page_to_pvec(&page, "iasbuilt-server", "0.0.0").unwrap_err();
    assert_eq!(
        error.to_string(),
        "The PDF has no supported vector paths for PlanViewer."
    );
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}
