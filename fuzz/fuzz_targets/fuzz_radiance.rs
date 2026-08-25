#![no_main]

use libfuzzer_sys::fuzz_target;
use libviprs::source::DecodeLimits;

/// Fuzz the hand-rolled Radiance `.hdr` decoder.
///
/// Radiance is the one raster container libviprs parses itself rather than
/// through the `image` facade, so it owns the header scan, the resolution
/// line, and both run-length scanline decoders. `fuzz_decode` reaches the
/// same code through the sniff table, but this target drives it directly so
/// the seeded corpus in `corpus/fuzz_radiance/` stays meaningful: every
/// seed there is a specific malformation the decoder has to reject with a
/// typed error rather than a panic, an over-read, or an unbounded loop.
///
/// The allocation budget is deliberately small. A `.hdr` body is run-length
/// encoded, so a handful of bytes can declare a very large image, and a
/// fuzzer will find that in seconds.
fuzz_target!(|data: &[u8]| {
    let limits = DecodeLimits::default().with_max_alloc_bytes(4 * 1024 * 1024);
    let _ = libviprs::decode_radiance(data, limits);
});
