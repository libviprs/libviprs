//! Every bound in `ProviderLimits::hardened()` is shown to bite (issues #1391, #1256).
//!
//! `max_entities` and `max_input_bytes` already had a test. The other four were
//! set on the native reader and never exercised, which is how a limit stops
//! working without anyone noticing: a field renamed in the mapping, a default
//! that quietly wins. Each test here decodes a committed drawing that holds
//! what the field counts, twice: once with the bound under that amount, which
//! has to end the decode with the reader's own "hit a bound" diagnostic and no
//! drawing, and once with the hardened value, which has to deliver it all.
//!
//! This is the field-by-field proof. It does not show that a drawing built to
//! exceed the real hardened values (a block bomb, a 100 MB string, tens of
//! millions of points) is refused: those need the generator work in
//! libviprs-dep#162, and #1256 stays open for them.
#![cfg(feature = "dwg-acadsharp")]

use std::path::Path;

use libviprs::cad::acadsharp::{AcadSharpDecoder, ProviderLimits};
use libviprs::cad::{CadDecoder, CadSource, CollectSink, DecodeReport, ViewKind};

struct Outcome {
    ok: bool,
    primitives: usize,
    hit_a_bound: bool,
}

fn decode(file: &str, limits: ProviderLimits) -> Option<Outcome> {
    let decoder = match AcadSharpDecoder::new() {
        Ok(d) => d.with_limits(limits),
        Err(e) => {
            assert!(
                std::env::var("LIBVIPRS_REQUIRE_ACADSHARP_NATIVE").as_deref() != Ok("1"),
                "the native lane was required and the decoder did not start: {e:?}"
            );
            eprintln!("skipped: no native acadsharp library linked ({e})");
            return None;
        }
    };
    let bytes = std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/cad/g13")
            .join(file),
    )
    .unwrap();
    let drawing = decoder.open(CadSource::Bytes(&bytes)).unwrap();
    let model = drawing
        .views()
        .unwrap()
        .into_iter()
        .find(|v| v.kind() == ViewKind::Model)
        .unwrap();
    let mut sink = CollectSink::new();
    let mut report = DecodeReport::new();
    let r = drawing.decode(model.index(), &mut sink, &mut report);
    Some(Outcome {
        ok: r.is_ok(),
        primitives: sink.len(),
        hit_a_bound: report
            .diagnostics()
            .iter()
            .any(|d| d.message().contains("hit a bound")),
    })
}

/// The bound under the amount the drawing holds ends the decode; the hardened
/// value delivers the drawing. Both halves matter: the second is what shows the
/// first failure came from the bound and not from the drawing.
fn bites(file: &str, tight: ProviderLimits, delivered: usize) {
    let hardened = ProviderLimits::hardened();
    let Some(free) = decode(file, hardened) else {
        return;
    };
    assert!(
        free.ok && free.primitives >= delivered && !free.hit_a_bound,
        "{file} should decode under the hardened limits: ok={} prims={}",
        free.ok,
        free.primitives
    );
    let held = decode(file, tight).unwrap();
    assert!(
        !held.ok && held.hit_a_bound,
        "{file} under {tight:?} should end on a bound: ok={} bound={}",
        held.ok,
        held.hit_a_bound
    );
    assert!(
        held.primitives < free.primitives,
        "{file} under {tight:?} still delivered {} of {}",
        held.primitives,
        free.primitives
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn max_string_bytes_stops_a_decode_with_a_long_string() {
    bites(
        "g13_long_text.dwg",
        ProviderLimits {
            max_string_bytes: Some(1),
            ..ProviderLimits::hardened()
        },
        1,
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn max_polyline_points_stops_a_decode_with_a_long_polyline() {
    bites(
        "g13_polyline.dwg",
        ProviderLimits {
            max_polyline_points: Some(1),
            ..ProviderLimits::hardened()
        },
        3,
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn max_block_depth_stops_a_decode_with_nested_blocks() {
    bites(
        "g13_deep_blocks.dwg",
        ProviderLimits {
            max_block_depth: Some(1),
            ..ProviderLimits::hardened()
        },
        1,
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn max_output_bytes_stops_a_decode_that_emits_a_lot() {
    bites(
        "g13_many_inserts.dwg",
        ProviderLimits {
            max_output_bytes: Some(1000),
            ..ProviderLimits::hardened()
        },
        30_000,
    );
}

/// The supervisor pattern is one thing a deployer reads, in the provider docs
/// that the CLI help and the README point at (#1256).
#[test]
#[cfg_attr(miri, ignore)]
fn the_provider_docs_carry_the_worker_pattern() {
    let docs = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("docs/vector-providers.md"),
    )
    .unwrap();
    for needle in [
        "the worker is the sandbox",
        "ProviderLimits::hardened()",
        "ulimit -v",
        "docker run --memory=256m",
        "TilerConfig::deadline",
        "<dest>.tmp.data",
        "<dest>.tmp.idx",
    ] {
        assert!(
            docs.contains(needle),
            "docs/vector-providers.md no longer names {needle}"
        );
    }
}
