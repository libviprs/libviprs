//! libviprs/pvec#302 (A2.2): what `pdf_page_to_pvec` writes stays inside what the paint driver works out.
//! The page is `blueprint-mix.pdf`. The feature keys are read from the component streams the writer put in
//! the file (`pvec_scene::features_used`), so there is no mapping from `FillConstruct` or `ReasonCode` here
//! that could drift, and the test pins no bytes, only the feature set. Needs `--features pvec-write`.

#![cfg(feature = "pvec-write")]

use libviprs::pvec::{PdfVectorLimits, extract_page_paths, pdf_page_to_pvec};
use pvec_facade::PvecReader;
use pvec_scene::{SUPPORTED, features_used};
use std::path::PathBuf;

fn blueprint_mix_features() -> Option<Vec<&'static str>> {
    let fixtures = std::env::var_os("LIBVIPRS_TESTS_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../libviprs-tests/tests/fixtures")
        });
    let path = fixtures.join("blueprint-mix.pdf");
    if !path.is_file() {
        eprintln!(
            "skipped: {} is not there (set LIBVIPRS_TESTS_FIXTURES)",
            path.display()
        );
        return None;
    }
    let page = extract_page_paths(&path, 1, &PdfVectorLimits::default()).unwrap();
    let bytes = pdf_page_to_pvec(&page, "test", "0").unwrap();
    let (streams, _) = PvecReader::open(bytes).unwrap().streams().unwrap();
    Some(features_used(&streams))
}

#[test]
#[cfg_attr(miri, ignore)]
fn blueprint_mix_writes_only_features_the_driver_supports() {
    let Some(used) = blueprint_mix_features() else {
        return;
    };
    assert!(
        used.contains(&"23.stroke_item"),
        "the probe read almost nothing from the written streams: {used:?}"
    );
    let outside: Vec<&&str> = used.iter().filter(|k| !SUPPORTED.contains(*k)).collect();
    assert!(
        outside.is_empty(),
        "pdf_page_to_pvec wrote {outside:?}, which pvec-scene::SUPPORTED does not list; either the driver learns \
         it (docs/features.toml in libviprs/pvec) or the writer stops emitting it"
    );
}
