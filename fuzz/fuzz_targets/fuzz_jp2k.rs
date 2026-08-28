#![no_main]

use libfuzzer_sys::fuzz_target;
use libviprs::source::DecodeLimits;

/// Fuzz the JPEG 2000 decoder.
///
/// Two surfaces meet here and only one of them is a dependency.
/// `hayro-jpeg2000` is `#![forbid(unsafe_code)]`, so what a fuzzer finds in it
/// is a panic or an unbounded allocation rather than memory unsafety. The
/// other surface is `crate::jp2k`'s own, and it is the reason this target
/// exists: the module walks the ISO/IEC 15444-1 box structure and the `SIZ` /
/// `COD` marker segments by hand, for four things the decoder does not report
/// (the per-component sign bit, the subsampling factors, the tile geometry and
/// the `METH = 2` ICC payload), and every one of those reads is an
/// attacker-controlled length feeding a slice index.
///
/// The seeds in `corpus/fuzz_jp2k/` are shaped around that. The `nocrash-`
/// ones are hand-built box and marker headers whose lengths lie: shorter than
/// their own header, past the end of the file, zero when zero means "to the
/// end", an extended `XLBox` that does not fit in a `usize`, a marker segment
/// whose length is under the length field itself. The `rejected-` ones are the
/// oracle capture's malformed fixtures, each cut at a structural boundary. The
/// `valid-` ones are there so the fuzzer starts from bytes that reach the
/// decoder at all, and cover both containers, both element widths, the
/// subsampled path, the tiled path and the multi-resolution path.
///
/// It drives `decode_jp2k` directly rather than going through the sniff table,
/// so a seed that stops matching the magic stays meaningful rather than
/// silently testing nothing.
///
/// The allocation budget is deliberately small. A `SIZ` marker can declare a
/// very large image in forty bytes and a fuzzer will find that in seconds; the
/// budget is also what this checks reaches the decoder, since the module
/// prices the component buffers from the declared geometry before the decoder
/// reserves anything.
fuzz_target!(|data: &[u8]| {
    let limits = DecodeLimits::default().with_max_alloc_bytes(4 * 1024 * 1024);
    let _ = libviprs::decode_jp2k(data, limits);
});
