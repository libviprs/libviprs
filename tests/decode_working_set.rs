//! What a decode really allocates, against what it priced (issue #944).
//!
//! [`DecodeLimits::max_alloc_bytes`] is documented as "the maximum number of
//! bytes the decoder may allocate at one time". #892 established what that
//! costs to mean: "the WebP decode budget covers what `image-webp` allocates,
//! not only what libviprs fills". Three decoders in the same window kept
//! pricing only the raster, so a caller sizing a container limit from
//! `max_alloc_bytes` was killed at up to 6.45x it, and the refusal message
//! understated by the same factor, which sends anyone tuning from the error
//! to the wrong number.
//!
//! # How the price is read
//!
//! Not restated here. Each case decodes once with `max_alloc_bytes = 1`,
//! which every one of these decoders refuses before it reserves a frame, and
//! takes the `needed_bytes` out of the refusal. That is the crate's own
//! number, arrived at the way a caller would arrive at it, so this file
//! cannot pass by agreeing with a model it also wrote. The label the refusal
//! carries is asserted too, so a different ceiling firing first would fail
//! rather than quietly hand over the wrong price.
//!
//! # Why this is its own binary
//!
//! It installs a counting global allocator, and a crate gets one of those.
//! `tests/webp_decode_working_set.rs` is the sibling #892 left; this one
//! carries the WebP case again as its **control**, because a harness that
//! reports over-one for every decoder proves nothing. WebP has priced its
//! decoder's planes since #892 and reads under one here.
//!
//! # Why 512x512
//!
//! The same argument that file makes. The thing under test is asymptotic: at
//! 4x4 a decoder's fixed overheads (chunk maps, bit readers, `Vec` growth)
//! dwarf the pixels, and `peak <= priced` would be false for reasons that
//! have nothing to do with pricing. At 512x512 the planes are the budget.
//!
//! # The AVIF fixtures
//!
//! AVIF is load-only here, so its cases are committed files rather than
//! encoded rasters, written by the pinned oracle from a smooth 512x512 ramp:
//!
//! ```text
//! vips heifsave ramp.png rgb444_512.avif  --compression av1 --bitdepth 8 \
//!     --Q 50 --effort 0 --subsample-mode off --keep none
//! vips heifsave ramp.png rgb420_512.avif  --compression av1 --bitdepth 8 \
//!     --Q 50 --effort 0 --subsample-mode on  --keep none
//! vips heifsave rampa.png rgba444_512.avif --compression av1 --bitdepth 8 \
//!     --Q 50 --effort 0 --subsample-mode off --keep none
//! ```
//!
//! Four rather than one because the AVIF working set is the shape of the
//! decoded frame, so subsampling and the alpha item are exactly the two terms
//! that move: 4:4:4 keeps three full planes, 4:2:0 keeps one and two quarters,
//! and an alpha item is a second frame held while the first is assembled. A
//! single 4:4:4 fixture would leave both untested, and the 4:2:0-with-alpha
//! one is there because at 4:4:4 the alpha term can be dropped and the bound
//! still holds.
//!
//! ```text
//! vips heifsave rampa.png rgba420_512.avif --compression av1 --bitdepth 8 \
//!     --Q 50 --effort 0 --subsample-mode on  --keep none
//! ```
//!
//! Every fixture arrives through `include_bytes!`, so nothing here touches
//! the filesystem at run time and no row belongs in
//! `tests/miri_fs_test_inventory.txt`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use libviprs::source::{DecodeLimits, decode_bytes_with_limits};
use libviprs::{PixelFormat, Raster, SourceError};

#[path = "common/working_set_coverage.rs"]
mod coverage;

/// 4:4:4 8-bit, no alpha: three full-size planes, the widest frame an 8-bit
/// AVIF can decode to.
const AVIF_444: &[u8] = include_bytes!("fixtures/rgb444_512.avif");
/// 4:2:0 8-bit: one full plane and two quarter planes, so the chroma term of
/// the price has to follow `av1C` rather than assume the worst.
const AVIF_420: &[u8] = include_bytes!("fixtures/rgb420_512.avif");
/// 4:4:4 8-bit with an alpha item, which is a **second** decoded frame held
/// alongside the first while `assemble` runs.
const AVIF_444A: &[u8] = include_bytes!("fixtures/rgba444_512.avif");
/// 4:2:0 8-bit with an alpha item, which is the case that makes the alpha
/// term of the price load bearing.
///
/// At 4:4:4 the primary frame is three full planes and the slack in that term
/// happens to cover the alpha frame as well, so dropping the alpha term does
/// not break the bound. At 4:2:0 the primary frame is one and a half planes
/// and the alpha frame is a full one, so the term is the difference between
/// bounding this decode and not.
const AVIF_420A: &[u8] = include_bytes!("fixtures/rgba420_512.avif");

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

/// A 512x512 ramp in `bands` bands, as the source for the encodable cases.
///
/// The values are a ramp rather than a constant because a constant image
/// compresses to almost nothing and some of these decoders size a buffer from
/// what they read; the allocation under test is driven by the geometry either
/// way, and the ramp keeps the codestreams a realistic size.
fn ramp(dim: u32, bands: u32) -> Raster {
    let n = (dim * dim * bands) as usize;
    let format = if bands == 4 {
        PixelFormat::Rgba8
    } else {
        PixelFormat::Rgb8
    };
    Raster::new(dim, dim, format, (0..n).map(|v| (v / 7) as u8).collect()).expect("ramp fixture")
}

/// One container, at a geometry big enough for its planes to be the budget.
struct Case {
    /// The name the coverage list and the refusal-shape table know it by.
    format: &'static str,
    /// The `what` the frame refusal must carry. Asserted, so a different
    /// ceiling firing first is a failure rather than a wrong price.
    label: &'static str,
    bytes: Vec<u8>,
}

fn cases() -> Vec<Case> {
    let mut cases = vec![
        // The control: priced since #892, so it reads under one.
        Case {
            format: "webp",
            label: "WebP frame buffer",
            bytes: ramp(512, 3).encode_webp(Default::default()).expect("webp"),
        },
        Case {
            format: "gif",
            label: "GIF canvas",
            bytes: ramp(512, 3).encode_gif(Default::default()).expect("gif"),
        },
    ];
    if cfg!(feature = "avif") {
        for bytes in [AVIF_444, AVIF_420, AVIF_444A, AVIF_420A] {
            cases.push(Case {
                format: "avif",
                label: "AVIF frame buffer",
                bytes: bytes.to_vec(),
            });
        }
    }
    if cfg!(feature = "jp2k") {
        cases.push(Case {
            format: "jp2k",
            label: "JPEG 2000 component buffers",
            bytes: ramp(512, 3).encode_jp2k(Default::default()).expect("jp2k"),
        });
    }
    cases
}

/// What the decoder charges for `case`, taken from its own refusal rather
/// than restated.
fn priced(case: &Case) -> u64 {
    let err =
        decode_bytes_with_limits(&case.bytes, DecodeLimits::default().with_max_alloc_bytes(1))
            .expect_err("a one-byte budget must refuse every case here");
    let SourceError::AllocLimitExceeded {
        what, needed_bytes, ..
    } = err
    else {
        panic!("{}: not the shared refusal shape: {err:?}", case.format);
    };
    assert_eq!(
        what, case.label,
        "{}: a ceiling other than the frame price fired first, so needed_bytes \
         is not the frame price",
        case.format
    );
    needed_bytes
}

/// The peak live bytes a clean decode of `case` reaches.
fn peak(case: &Case) -> i64 {
    let (decoded, peak) = peak_during(|| {
        decode_bytes_with_limits(
            &case.bytes,
            DecodeLimits::default().with_max_alloc_bytes(u64::MAX),
        )
    });
    let decoded = decoded
        .unwrap_or_else(|e| panic!("{}: must decode with the budget lifted: {e}", case.format));
    assert_eq!(
        (decoded.width(), decoded.height()),
        (512, 512),
        "{}: decoded at a geometry this file does not claim",
        case.format
    );
    peak
}

/// Issue #944. The amount a decoder prices against `max_alloc_bytes` has to
/// be an upper bound on what the decode actually allocates, or the ceiling
/// bounds nothing and the number in the refusal message is not the number a
/// caller should raise it to.
///
/// Every case is reported rather than the first failure, because the ratios
/// are the finding: three decoders at 6.45x, 2.35x and 2.00x is a different
/// story from one of them being slightly out.
#[test]
fn the_price_bounds_what_the_decode_allocates() {
    let mut over = Vec::new();
    for case in cases() {
        let priced = priced(&case);
        let peak = peak(&case);
        if peak as u64 > priced {
            over.push(format!(
                "{}: priced {priced}, peaked at {peak}, {:.2}x",
                case.format,
                peak as f64 / priced as f64
            ));
        }
    }
    assert!(
        over.is_empty(),
        "{} decoder(s) allocate more than max_alloc_bytes lets them say they \
         will, so a caller sizing a container limit from the refusal is killed \
         at that factor (issue #944):\n  {}",
        over.len(),
        over.join("\n  ")
    );
}

/// Issue #944, the other side. A price that is an upper bound because it is
/// enormous refuses files that fit, which is the failure mode the WebP fix
/// traded into (it prices at 2x its measured peak). Nothing here may go past
/// twice the peak.
#[test]
fn the_price_is_not_far_more_than_the_peak() {
    let mut loose = Vec::new();
    for case in cases() {
        let priced = priced(&case);
        let peak = peak(&case);
        if priced > (peak as u64).saturating_mul(2) {
            loose.push(format!(
                "{}: priced {priced} against a peak of {peak}, {:.2}x",
                case.format,
                priced as f64 / peak as f64
            ));
        }
    }
    assert!(
        loose.is_empty(),
        "{} decoder(s) price more than twice what they spend, which refuses \
         files that would have fitted:\n  {}",
        loose.len(),
        loose.join("\n  ")
    );
}

/// Issue #944. The coverage list may not claim a measurement this binary does
/// not make.
///
/// Both directions, because each catches the opposite lie: an entry naming
/// this binary with no case behind it is a container nobody measured, and a
/// case here that the list does not mention is a measurement
/// `tests/decode_alloc_refusal_shape.rs` will not credit, so its row would
/// have to sit on the unmeasured list while the work was already done.
#[test]
fn this_binary_runs_exactly_the_cases_the_coverage_list_assigns_it() {
    let mut ran: Vec<&str> = cases().iter().map(|c| c.format).collect();
    ran.sort_unstable();
    ran.dedup();

    let mut assigned: Vec<&str> = coverage::measured()
        .into_iter()
        .filter(|(_, by)| *by == "decode_working_set")
        .map(|(format, _)| format)
        .collect();
    assigned.sort_unstable();

    assert_eq!(
        ran, assigned,
        "the cases this binary runs and the containers the coverage list \
         assigns it must be the same set"
    );
    assert!(
        !ran.is_empty(),
        "an empty case list would satisfy the comparison above without \
         measuring anything"
    );
}
