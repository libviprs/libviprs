#![no_main]

use libfuzzer_sys::fuzz_target;
use libviprs::source::DecodeLimits;

/// Fuzz the OpenEXR loader.
///
/// EXR is a compressed, attribute-driven container: a few hundred bytes of
/// header can declare a gigapixel data window, an arbitrary channel list,
/// a tile grid, and any of ten compression codecs, each with its own
/// entropy decoder. `fuzz_decode` reaches the same code through the sniff
/// table, but this target drives it directly so the seeded corpus in
/// `corpus/fuzz_exr/` stays meaningful: every seed there is a specific
/// shape the loader has to answer with a typed error rather than a panic,
/// an over-read, or an unbounded allocation.
///
/// The allocation budget is deliberately small. The header declares the
/// geometry and the body is compressed, so a fuzzer finds the
/// small-file-huge-window shape in seconds.
fuzz_target!(|data: &[u8]| {
    let limits = DecodeLimits::default().with_max_alloc_bytes(4 * 1024 * 1024);
    let _ = libviprs::decode_exr(data, limits);
});
