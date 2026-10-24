//! `default_decoder(hardened)` applies the hardened provider limits only when
//! asked (issue #1292).
//!
//! The decoder comes back as a `Box<dyn CadDecoder>`, so a test cannot read its
//! limits. What it can see is behaviour only the limits produce: a file larger
//! than the hardened input cap is refused before it is read when `hardened` is
//! set, and is let through to the parser when it is not. A sparse file stands
//! in for the big drawing, so it costs no disk.
//!
//! These cells need the native ACadSharp library. Without one they skip, unless
//! `LIBVIPRS_REQUIRE_ACADSHARP_NATIVE=1` makes that a failure.
#![cfg(feature = "dwg-acadsharp")]

use std::io::Write;

use libviprs::cad::acadsharp::ProviderLimits;
use libviprs::cad::{CadDecoder, CadError, CadSource};
use libviprs::vector::convert::default_decoder;

/// A DWG version stamp followed by zeros, one byte past the hardened input cap.
fn oversized_drawing(dir: &std::path::Path) -> std::path::PathBuf {
    let cap = ProviderLimits::hardened().max_input_bytes.unwrap();
    let path = dir.join("huge.dwg");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(b"AC1032").unwrap();
    f.set_len(cap + 1).unwrap();
    path
}

fn decoder(hardened: bool) -> Option<Box<dyn CadDecoder>> {
    match default_decoder(hardened) {
        Ok(d) => Some(d),
        Err(e) => {
            assert!(
                std::env::var("LIBVIPRS_REQUIRE_ACADSHARP_NATIVE").as_deref() != Ok("1"),
                "native lane required: {e}"
            );
            eprintln!("skipped: {e}");
            None
        }
    }
}

fn why(result: Result<impl Sized, CadError>) -> String {
    match result {
        Ok(_) => "opened".to_owned(),
        Err(e) => format!("{e:?}"),
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_hardened_decoder_refuses_a_file_past_the_input_cap() {
    let Some(d) = decoder(true) else { return };
    let dir = tempfile::tempdir().unwrap();
    let path = oversized_drawing(dir.path());
    let err = why(d.open(CadSource::Path(&path)).map(|_| ()));
    assert_ne!(err, "opened", "a file past the hardened cap was opened");
    assert!(
        err.contains("LimitExceeded"),
        "the refusal should be the provider's limit error: {err}"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn the_plain_decoder_does_not_apply_the_input_cap() {
    let Some(d) = decoder(false) else { return };
    let dir = tempfile::tempdir().unwrap();
    let path = oversized_drawing(dir.path());
    let err = why(d.open(CadSource::Path(&path)).map(|_| ()));
    // The zeros are not a drawing, so this is refused too, but by the parser
    // and not by the cap.
    assert!(
        !err.contains("LimitExceeded"),
        "the plain decoder applied a limit it was not given: {err}"
    );
}

/// The numbers the documentation promises, so they cannot move quietly.
#[test]
fn the_hardened_limits_are_the_documented_ones() {
    let l = ProviderLimits::hardened();
    assert_eq!(l.max_input_bytes, Some(256 << 20));
    assert_eq!(l.max_entities, Some(10_000_000));
    assert_eq!(l.max_string_bytes, Some(1 << 20));
    assert_eq!(l.max_polyline_points, Some(1_000_000));
    assert_eq!(l.max_block_depth, Some(64));
    assert_eq!(l.max_output_bytes, Some(2 << 30));
    assert_eq!(
        ProviderLimits::default(),
        ProviderLimits {
            max_input_bytes: None,
            max_entities: None,
            max_string_bytes: None,
            max_polyline_points: None,
            max_block_depth: None,
            max_output_bytes: None,
        }
    );
}
