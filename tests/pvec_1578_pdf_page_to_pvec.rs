//! Issue #1578 (the L4 mapping, the part that needs none of the new streams; L1 to L3 and L5 to
//! L7 stay open): `pvec::pdf_page_to_pvec` is the mapping iasbuilt/server main's `convert_pdf`
//! runs since server #336 (white fills dropped, closed fills as polygons, rectangular clips
//! applied), moved into the library. The bytes it writes for `blueprint-mix.pdf` are pinned to
//! what server main's `convert_pdf` writes for the same page, producer `iasbuilt-server` 1.0.3
//! (server commit 33e48f5). Needs `--features pvec-write`.

#![cfg(feature = "pvec-write")]

use libviprs::pvec::{PdfVectorLimits, extract_page_paths, pdf_page_to_pvec};
use sha2::{Digest, Sha256};
use std::path::PathBuf;

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
#[cfg_attr(miri, ignore)]
fn blueprint_mix_bytes_are_pinned_with_the_stroke_streams() {
    let Some(path) = blueprint_mix() else { return };
    let page = extract_page_paths(&path, 1, &PdfVectorLimits::default()).unwrap();
    let bytes = pdf_page_to_pvec(&page, "iasbuilt-server", "1.0.3").unwrap();
    assert_eq!(bytes.len(), 16400);
    let hex: String = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(
        hex,
        "3e94b62fd975928c65d38b875d1ab9e41aa27abd0b4b32e3019e77bed3c4855c"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
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
