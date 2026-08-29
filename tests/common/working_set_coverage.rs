//! Which containers have had their decode-time **working set** measured
//! against the amount they price, and which have not (issue #944).
//!
//! `DecodeLimits::max_alloc_bytes` is documented as "the maximum number of
//! bytes the decoder may allocate at one time", so a price that covers only
//! the raster libviprs fills is not a ceiling at all for a decoder that keeps
//! buffers of its own beside it. #892 found that for WebP and fixed it there.
//! Three more decoders had the same shape and nobody had looked, because
//! `tests/decode_alloc_refusal_shape.rs` grew a column for it and left the
//! three at zero.
//!
//! This list is what stops that happening again. Every row of that file's
//! self-priced table has to appear in exactly one of the two lists below, so
//! a new container cannot be added with an unexamined working set: it either
//! gets a case in `tests/decode_working_set.rs`, which decodes it under a
//! counting global allocator, or it goes in [`unmeasured`] with a sentence
//! saying why not.
//!
//! Both lists are consumed from two directions, which is what keeps them
//! honest. `tests/decode_working_set.rs` asserts the cases it runs are
//! **exactly** the entries pointing at it, so an entry cannot claim a
//! measurement that does not exist. `tests/decode_alloc_refusal_shape.rs`
//! asserts every row it prices appears in exactly one list, so a row cannot
//! sit outside both.

#![allow(dead_code)]

/// Containers whose peak allocation is measured against their price, and the
/// test binary that measures it.
///
/// Feature-gated rows are absent from a build that cannot decode them, which
/// is the same shape the refusal-shape table's rows have, so the two sets line
/// up in every build rather than only in the all-features one.
pub fn measured() -> Vec<(&'static str, &'static str)> {
    let mut rows = vec![
        // The control. WebP has priced its decoder's working set since #892,
        // so it is the row that reads **under** one, and it is what says the
        // harness can tell a bounded decode from an unbounded one rather than
        // reporting over-one for everything.
        ("webp", "decode_working_set"),
        ("gif", "decode_working_set"),
    ];
    if cfg!(feature = "avif") {
        rows.push(("avif", "decode_working_set"));
    }
    if cfg!(feature = "jp2k") {
        rows.push(("jp2k", "decode_working_set"));
    }
    rows
}

/// Containers whose working set is still a claim rather than a measurement,
/// each with the reason it is not in [`measured`].
///
/// The goal is an empty list. Nothing here is asserted to be correct; the
/// point is that the claim is **visible**, where `decoder_planes: 0` was not.
pub fn unmeasured() -> Vec<(&'static str, &'static str)> {
    let mut rows = vec![
        // These five decode inside libviprs, out of a buffer the module
        // itself sized: the bytes priced are the bytes allocated, and there
        // is no second crate holding planes beside them. That is an argument
        // rather than a measurement, which is why they are on this list.
        ("v", "decoded in-crate, straight out of the file's own byte range"),
        ("ppm", "decoded in-crate from the ASCII or binary sample run"),
        ("fits", "decoded in-crate from the raw data unit"),
        ("nifti", "decoded in-crate from the raw voxel array"),
        ("mat", "decoded in-crate from the (possibly inflated) array element"),
        // And these three have no way to reach an asymptotic geometry from
        // this crate: there is no encoder for them here, and every committed
        // fixture is a few pixels across, where a decoder's fixed overheads
        // dwarf the planes and the measurement says nothing (the argument
        // `tests/webp_decode_working_set.rs` makes for its 512x512 fixture).
        (
            "openexr",
            "no encoder here and the committed fixture is 8x4, too small to measure asymptotically",
        ),
        (
            "uhdr",
            "the only fixture is `uhdr::smallest_container()`, 8x8",
        ),
        (
            "radiance",
            "encodable, but the decode is in-crate and nothing else has been measured for it",
        ),
    ];
    if cfg!(feature = "jxl") {
        // `jxl-oxide` takes the caller's budget through `AllocTracker`
        // (`src/jxl.rs`), so the decoder refuses itself rather than being
        // bounded from outside. `tests/jxl_frame_price.rs` covers the price;
        // the tracker covers the working set.
        rows.push((
            "jxl",
            "the decoder carries the caller's budget in its own AllocTracker",
        ));
    }
    rows
}
