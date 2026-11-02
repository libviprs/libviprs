#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "pvec_bodies/spool.rs"]
mod body;

// The body is shared with `tests/pvec_1070_spool_fuzz.rs`, which runs the seed corpus through it.
fuzz_target!(|data: &[u8]| {
    body::run(data);
});
