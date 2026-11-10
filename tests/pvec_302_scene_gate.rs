//! libviprs/pvec#302 (A2.2): what `pdf_page_to_pvec` writes stays inside what the paint driver works out or the viewer consumes.
//! The page is `blueprint-mix.pdf`. The feature keys are read from the component streams the writer put in
//! the file (`pvec_scene::features_used`), so there is no mapping from `FillConstruct` or `ReasonCode` here
//! that could drift, and the test pins no bytes, only the feature set. A key passes when it is in `SUPPORTED` (the driver) or `CONSUMED` (the viewer reads it from the stream). Needs `--features pvec-write`.

#![cfg(feature = "pvec-write")]

use libviprs::pvec::{PdfVectorLimits, extract_page_paths, pdf_page_to_pvec};
use pvec_facade::PvecReader;
use pvec_facade::components::{ClipStream, FillPaintStream, PaintOrderStream, StrokePaintStream};
use pvec_scene::{CONSUMED, SUPPORTED, features_used};
use std::path::PathBuf;

#[test]
#[cfg_attr(miri, ignore)]
fn blueprint_mix_writes_only_features_the_driver_supports() {
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
        return;
    }
    let page = extract_page_paths(&path, 1, &PdfVectorLimits::default()).unwrap();
    let bytes = pdf_page_to_pvec(&page, "test", "0").unwrap();
    let (streams, _) = PvecReader::open(bytes).unwrap().streams().unwrap();
    // `features_used` drops a stream it cannot decode without saying so, so a bad stream would only show up as
    // a smaller feature set. Decode every paint stream the writer put in the file here, so that fails loudly.
    for s in &streams {
        let decoded = match s.capability {
            PaintOrderStream::CAPABILITY => PaintOrderStream::decode(&s.payload).map(drop),
            StrokePaintStream::CAPABILITY => StrokePaintStream::decode(&s.payload).map(drop),
            FillPaintStream::CAPABILITY => FillPaintStream::decode(&s.payload).map(drop),
            ClipStream::CAPABILITY => ClipStream::decode(&s.payload).map(drop),
            _ => Ok(()),
        };
        assert!(
            decoded.is_ok(),
            "stream {} the writer produced does not decode: {:?}",
            s.capability,
            decoded.err()
        );
    }
    let used = features_used(&streams);
    assert!(
        used.contains(&"23.stroke_item"),
        "the probe read almost nothing from the written streams: {used:?}"
    );
    let outside: Vec<&&str> = used
        .iter()
        .filter(|k| !SUPPORTED.contains(*k) && !CONSUMED.contains(*k))
        .collect();
    assert!(
        outside.is_empty(),
        "pdf_page_to_pvec wrote {outside:?}, which neither pvec-scene::SUPPORTED nor CONSUMED lists; either the driver or the viewer learns \
         it (docs/features.toml in libviprs/pvec) or the writer stops emitting it"
    );
}
