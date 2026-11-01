#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "pvec_bodies/provider.rs"]
mod body;

// The body is shared with `tests/pvec_1437_fuzz_seeds.rs`, which runs the seed corpus through it.
fuzz_target!(|data: &[u8]| body::run(data));
