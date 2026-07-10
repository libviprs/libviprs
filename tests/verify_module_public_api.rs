//! The `verify` and `stream_verify` modules are declared `pub mod` in the
//! crate root, and their module docs advertise `raster_verify` /
//! `verify_from_strip_source` as verify-mode entry points. This test compiles
//! as an *external* crate, so it can only name items that are genuinely part
//! of the public API — before issue #144 both entry points were `pub(crate)`,
//! leaving the public modules with nothing a downstream caller could reach,
//! and the very functions their docs linked to were unnameable from here.
//!
//! Each call below both (a) proves the public path resolves at compile time
//! and (b) exercises the real early-return contract: verify against a sink
//! with no on-disk checkpoint root must fail structurally with
//! [`EngineError::VerifyRequiresOnDiskSink`] rather than walking any tiles.

use libviprs::observe::NoopObserver;
use libviprs::{
    EngineConfig, EngineError, Layout, MemorySink, PixelFormat, PyramidPlanner, Raster,
    RasterStripSource,
};

fn tiny_raster() -> Raster {
    Raster::new(64, 64, PixelFormat::Rgb8, vec![0u8; 64 * 64 * 3]).unwrap()
}

/// `libviprs::verify::raster_verify` is reachable as public API and honours
/// the "verify needs an on-disk sink" contract.
#[test]
fn raster_verify_is_public_and_requires_on_disk_sink() {
    let src = tiny_raster();
    let plan = PyramidPlanner::new(64, 64, 32, 0, Layout::DeepZoom)
        .unwrap()
        .plan();
    let sink = MemorySink::new();
    let config = EngineConfig::default();
    let observer = NoopObserver;

    let result = libviprs::verify::raster_verify(&src, &plan, &sink, &config, &observer);

    assert!(
        matches!(result, Err(EngineError::VerifyRequiresOnDiskSink)),
        "raster_verify against a checkpoint-less sink must fail structurally, got {result:?}"
    );
}

/// `verify_from_strip_source` is reachable both through the `verify` façade
/// module and its home `stream_verify` module, and honours the same contract.
#[test]
fn verify_from_strip_source_is_public_and_requires_on_disk_sink() {
    let src = tiny_raster();
    let plan = PyramidPlanner::new(64, 64, 32, 0, Layout::DeepZoom)
        .unwrap()
        .plan();
    let strip = RasterStripSource::new(&src);
    let sink = MemorySink::new();
    let config = EngineConfig::default();
    let observer = NoopObserver;

    let via_facade =
        libviprs::verify::verify_from_strip_source(&strip, &plan, &sink, &config, &observer);
    assert!(
        matches!(via_facade, Err(EngineError::VerifyRequiresOnDiskSink)),
        "verify::verify_from_strip_source against a checkpoint-less sink must fail structurally, got {via_facade:?}"
    );

    let via_home =
        libviprs::stream_verify::verify_from_strip_source(&strip, &plan, &sink, &config, &observer);
    assert!(
        matches!(via_home, Err(EngineError::VerifyRequiresOnDiskSink)),
        "stream_verify::verify_from_strip_source against a checkpoint-less sink must fail structurally, got {via_home:?}"
    );
}
