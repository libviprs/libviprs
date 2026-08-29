//! The WebP decode budget has to cover what `image-webp` allocates, not only
//! what libviprs fills (issue #892).
//!
//! # Why this is its own binary
//!
//! It installs a counting global allocator, and a crate gets one of those.
//! Putting it here rather than in `src/webp.rs`'s unit tests keeps every
//! other test out of the counter's way and keeps the counter out of theirs.
//!
//! # Why the fixture is 512x512
//!
//! The thing under test is asymptotic. A 4x3 frame prices 36 bytes and peaks
//! at 2466, because the decoder's fixed overheads (its chunk map, its
//! bit-reader, `Vec` growth) dwarf the pixels; the plane model says nothing
//! at that size and `peak <= priced` would be false for a reason that has
//! nothing to do with pricing. At 512x512 the planes are the budget and the
//! slack lands on whole RGBA planes of one frame, which is what the
//! decoder's structure predicts.
//!
//! The fixture arrives through `include_bytes!`, so nothing here touches the
//! filesystem at run time and no row belongs in
//! `tests/miri_fs_test_inventory.txt`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use libviprs::source::DecodeLimits;
use libviprs::webp;

/// `img2webp -loop 0 -lossless -d 100 big0.png -d 100 big1.png`: two 512x512
/// full-canvas frames, no alpha, no disposal, both `VP8L`.
///
/// Written by libwebp's own tool because `vips webpsave` cannot be asked for
/// a two-frame animation at a chosen size without going through a roll, and
/// what this test needs is the geometry rather than any particular pixel.
const ANIM512: &[u8] = include_bytes!("fixtures/anim512.webp");

thread_local! {
    /// Live bytes and the high-water mark, per thread, because the harness
    /// runs tests in parallel and a shared counter would see every sibling's
    /// traffic. Both are `Cell`, which is read and written without
    /// allocating, so the allocator cannot recurse into itself.
    static LIVE: Cell<i64> = const { Cell::new(0) };
    static PEAK: Cell<i64> = const { Cell::new(0) };
}

fn bump(delta: i64) {
    LIVE.with(|live| {
        let now = live.get() + delta;
        live.set(now);
        PEAK.with(|peak| {
            if now > peak.get() {
                peak.set(now);
            }
        });
    });
}

/// The system allocator with a per-thread high-water mark in front of it.
struct Counting;

// SAFETY: every method forwards to `System` with the layout it was handed and
// returns exactly what `System` returned, so the allocator contract is
// whatever `System`'s is. The counter is two thread-local `Cell<i64>`s, read
// and written without allocating.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        bump(layout.size() as i64);
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        bump(layout.size() as i64);
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        bump(new_size as i64 - layout.size() as i64);
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        bump(-(layout.size() as i64));
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

/// Run `f` and report the highest number of live bytes it reached.
fn peak_during<T>(f: impl FnOnce() -> T) -> (T, i64) {
    LIVE.with(|live| live.set(0));
    PEAK.with(|peak| peak.set(0));
    let value = f();
    (value, PEAK.with(Cell::get))
}

/// Tests that the amount `decode_webp_with` prices against
/// `DecodeLimits::max_alloc_bytes` is an upper bound on what the decode
/// actually allocates, which is the whole point of having the ceiling. Works
/// by decoding a 512x512 two-frame animation under a counting global
/// allocator and comparing the high-water mark to the price.
///
/// Input: `ANIM512` at one page and at both -> Output: a peak at or under
/// the priced amount, with the headroom reported so a change in either
/// direction is visible in the failure message.
#[test]
fn the_priced_amount_bounds_what_the_decode_allocates() {
    for (label, n, pages) in [("one page", 1i32, 1i64), ("both pages", -1, 2)] {
        let (raster, peak) = peak_during(|| {
            webp::decode_webp_with(
                ANIM512,
                DecodeLimits::default(),
                webp::LoadOptions::default().with_n(n),
            )
        });
        let raster = raster.unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(raster.width(), 512);
        assert_eq!(i64::from(raster.height()), 512 * pages);

        // The formula the loader uses, restated here rather than imported,
        // so a change to it has to be made twice and the second time is a
        // sentence a reviewer reads.
        let frame_px = 512i64 * 512;
        let roll = frame_px * pages * i64::from(raster.format().channels() as u32);
        let priced = roll + webp::DECODER_PLANES_ANIMATED as i64 * frame_px * 4;

        assert!(
            peak <= priced,
            "{label}: the decode peaked at {peak} live bytes against a priced \
             {priced}, so the ceiling does not bound it"
        );
        // And the price is not absurd either: an order of magnitude of
        // headroom would mean the ceiling refuses files that fit.
        assert!(
            priced < peak * 2,
            "{label}: priced {priced} against a peak of {peak}, which is more \
             headroom than the model claims to need"
        );
    }
}

/// Tests that the still path is bounded too, on the same geometry, because
/// its working set is a different shape: one temporary RGBA plane it narrows
/// into the caller's buffer rather than a canvas plus a frame. Works by
/// re-encoding one page of the animation as a still and decoding that.
///
/// Input: a 512x512 lossless still -> Output: a peak at or under the priced
/// amount.
#[test]
fn the_still_price_bounds_the_still_decode() {
    let page = webp::decode_webp_with(
        ANIM512,
        DecodeLimits::default(),
        webp::LoadOptions::default(),
    )
    .expect("one page of the animation decodes");
    let still = page
        .encode_webp(webp::SaveOptions::default())
        .expect("a 512x512 Rgb8 raster encodes");

    let (decoded, peak) = peak_during(|| webp::decode_webp(&still, DecodeLimits::default()));
    let decoded = decoded.expect("the still decodes");
    let frame_px = 512i64 * 512;
    let priced = frame_px * i64::from(decoded.format().channels() as u32)
        + webp::DECODER_PLANES_STILL as i64 * frame_px * 4;
    assert!(
        peak <= priced,
        "the still decode peaked at {peak} against a priced {priced}"
    );
}
