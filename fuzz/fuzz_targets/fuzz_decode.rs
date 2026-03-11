#![no_main]

use libfuzzer_sys::fuzz_target;

/// Fuzz the image decode path with arbitrary bytes.
///
/// This exercises the `image` crate's decoders (JPEG, PNG, TIFF) via
/// `decode_bytes`, looking for panics, OOM, or undefined behavior.
fuzz_target!(|data: &[u8]| {
    // We don't care about the result — just that it doesn't panic or crash.
    let _ = libviprs::source::decode_bytes(data);
});
