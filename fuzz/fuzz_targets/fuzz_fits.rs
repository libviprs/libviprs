#![no_main]

use libfuzzer_sys::fuzz_target;
use libviprs::source::DecodeLimits;

/// Fuzz the hand-rolled FITS decoder.
///
/// FITS is one of the containers libviprs parses itself rather than through
/// the `image` facade, so it owns the header-card scan, the walk over
/// header units, the carrier resolution, and the de-planarising sample
/// copy. `fuzz_decode` reaches the same code through the sniff table, but
/// this target drives it directly so the seeded corpus in
/// `corpus/fuzz_fits/` stays meaningful: every seed there is a specific
/// malformation the decoder has to reject with a typed error rather than a
/// panic, an over-read, or an unbounded loop.
///
/// The allocation budget is deliberately small. A FITS header states its
/// geometry in ASCII, so a few dozen bytes can claim a gigapixel image, and
/// a fuzzer will find that in seconds.
fuzz_target!(|data: &[u8]| {
    let limits = DecodeLimits::default().with_max_alloc_bytes(4 * 1024 * 1024);
    let _ = libviprs::decode_fits(data, limits);
});
