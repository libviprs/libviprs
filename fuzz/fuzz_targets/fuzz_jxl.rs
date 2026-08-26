#![no_main]

use libfuzzer_sys::fuzz_target;
use libviprs::source::DecodeLimits;

/// Fuzz the JPEG XL decoder.
///
/// This is the format where a fuzz target earns its keep most obviously.
/// `jxl-oxide` is the widest surface libviprs points at untrusted bytes: a
/// bitstream parser, an entropy coder, a modular predictor tree, a VarDCT
/// inverse transform and an ISOBMFF box reader, across thirteen crates. It
/// has already shipped one memory-safety advisory of exactly the kind a
/// fuzzer finds, GHSA-66m8-c62j-h6v5, where `FrameBuffer::new` multiplied
/// `width * height * channels` as unchecked `usize` and handed out
/// oversized slices from the undersized buffer that resulted. The floor in
/// `Cargo.toml` keeps that one out; this target is for the next one.
///
/// It drives `decode_jxl` directly rather than going through the sniff
/// table, so the seeds in `corpus/fuzz_jxl/` stay meaningful: every seed
/// there is a specific malformation the decoder has to reject with a typed
/// error rather than a panic, an over-read, or an unbounded allocation.
/// Both container forms are seeded, because JPEG XL is the only format in
/// the crate with two unrelated magics.
///
/// The allocation budget is deliberately small. A JPEG XL header can
/// declare a very large image in a handful of bytes, and a fuzzer will find
/// that in seconds; the budget is also what the target is checking reaches
/// the decoder at all, since it is enforced in two places (the crate's own
/// ceiling on the declared geometry, and `jxl-oxide`'s `AllocTracker`).
fuzz_target!(|data: &[u8]| {
    let limits = DecodeLimits::default().with_max_alloc_bytes(4 * 1024 * 1024);
    let _ = libviprs::decode_jxl(data, limits);
});
