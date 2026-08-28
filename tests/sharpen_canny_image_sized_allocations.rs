//! Holds the "no image-sized infallible allocation left on this path" claim
//! that #669 wrote onto `try_sharpen` and `try_canny` (issue #700).
//!
//! Nothing held it. Two one-line mutations put an image-sized copy back with
//! the whole suite green: `let mut sharpened = labs.clone();` where the LabS
//! raster is moved into the result, and `let mut out_samples = samples.clone();`
//! where the sharpen curve is written back over the widened samples in place.
//! Both are exactly the copy #627 removed.
//!
//! No ceiling can catch either. The ceilings on these paths sit inside the
//! fallible reservation helpers, and a copy that goes through `Clone::clone` or
//! `Vec::clone` never reaches one; that is the whole reason the mutations are
//! invisible. So this asks the allocator instead, which does not care which
//! type is in front of it, and holds each path to the two numbers it measures
//! today.
//!
//! # The two numbers
//!
//! For each operation the guard counts, over one run:
//!
//! * **allocations** at or above one byte per pixel, which is every buffer that
//!   scales with the image and no fixed-size one (the Gaussian mask, the clamp
//!   tables and the per-row accumulator are all orders of magnitude below the
//!   threshold at the sizes used here);
//! * **peak live image-sized bytes per pixel**, the high-water mark of those
//!   buffers held at the same time, divided by the pixel count.
//!
//! Both are asserted as ceilings at the measured value, so there is no slack:
//! one more image-sized buffer does not fit under either.
//!
//! The second number is what makes the first one mean something. A count on its
//! own is a magic constant, and a reader has no way to tell a real budget from
//! a number someone fitted to make the test pass. A per-pixel figure is a
//! property of the algorithm, and the guard proves it is one by measuring every
//! row at **two different image sizes** and requiring the same count and the
//! same bytes per pixel from both. A constant fitted to one size does not
//! survive that.
//!
//! # What this cannot see
//!
//! * It cannot tell an image-sized allocation that went through a fallible
//!   reservation from one that did not. It is a budget, not a proof of
//!   fallibility. What it proves is that no *further* image-sized buffer fits,
//!   which is the property both mutations break.
//! * It only counts allocations made on the thread that runs the operation.
//!   Neither path spawns or fans out today, and the two-size check would catch
//!   a count that quietly stopped scaling, but work moved onto a worker thread
//!   would stop being charged here.
//! * It counts a request, not a residency. An allocator that over-allocates, or
//!   a page never touched, is charged at the size asked for.
//! * A change that legitimately adds an image-sized buffer to one of these
//!   paths reddens the row it belongs to, and the budget wants re-measuring
//!   with the same evidence that set it in the first place. That is the
//!   intended cost.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use libviprs::{ConvolutionError, Precision, Raster, generate_test_raster};

// --- counting global allocator ----------------------------------------------

thread_local! {
    /// Size at or above which an allocation is charged, in bytes. `usize::MAX`
    /// disarms, which is where every thread starts and where the measurement
    /// window leaves it, so tests running in parallel never see one another.
    static THRESHOLD: Cell<usize> = const { Cell::new(usize::MAX) };
    /// How many charged allocations this thread has made in the window.
    static COUNT: Cell<u32> = const { Cell::new(0) };
    /// Charged bytes currently live. Signed, because the window can outlive a
    /// buffer allocated before it opened and free is charged symmetrically.
    static LIVE: Cell<i64> = const { Cell::new(0) };
    /// The high-water mark `LIVE` reached in the window.
    static PEAK: Cell<i64> = const { Cell::new(0) };
}

/// Charge an allocation.
///
/// Every cell is a `Cell` of a `Copy` scalar with a `const` initialiser and no
/// destructor, so touching one allocates nothing and cannot recurse back into
/// this allocator. `try_with` covers the one case left, a thread whose
/// thread-local storage is already being torn down, by declining to charge
/// rather than panicking inside the allocator.
fn charge(size: usize) {
    let _ = THRESHOLD.try_with(|t| {
        if size >= t.get() {
            let _ = COUNT.try_with(|n| n.set(n.get() + 1));
            let _ = LIVE.try_with(|l| {
                let now = l.get() + size as i64;
                l.set(now);
                let _ = PEAK.try_with(|p| {
                    if now > p.get() {
                        p.set(now);
                    }
                });
            });
        }
    });
}

/// Release a charge. Deliberately not symmetric with [`charge`] on `COUNT`: a
/// buffer that is allocated and dropped inside the window still cost an
/// allocation and still counts as one.
fn release(size: usize) {
    let _ = THRESHOLD.try_with(|t| {
        if size >= t.get() {
            let _ = LIVE.try_with(|l| l.set(l.get() - size as i64));
        }
    });
}

struct CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            charge(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        release(layout.size());
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            charge(layout.size());
        }
        ptr
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
        if !new_ptr.is_null() {
            // A `Vec` that grows into image size made the same allocation a
            // `Vec` that reserved it did, just by a different route, so it is
            // charged on the new size and credited on the old.
            release(layout.size());
            charge(new_size);
        }
        new_ptr
    }
}

#[global_allocator]
static ALLOC: CountingAlloc = CountingAlloc;

/// What one run of an operation cost, in image-sized allocations and in peak
/// live image-sized bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Cost {
    allocs: u32,
    peak_bytes: i64,
}

/// Run `f` with this thread charging every allocation of `threshold` bytes or
/// more, and report what it cost.
///
/// The state is thread-local and restored on the way out, including on unwind,
/// so parallel tests and nested windows do not perturb one another. `f` returns
/// its value into the caller, which matters: the output raster is still alive
/// when the peak is read, so it is charged like every other buffer rather than
/// being freed out from under the measurement.
fn measure<R>(threshold: usize, f: impl FnOnce() -> R) -> (R, Cost) {
    struct Restore(usize, u32, i64, i64);
    impl Drop for Restore {
        fn drop(&mut self) {
            THRESHOLD.with(|t| t.set(self.0));
            COUNT.with(|n| n.set(self.1));
            LIVE.with(|l| l.set(self.2));
            PEAK.with(|p| p.set(self.3));
        }
    }
    let _restore = Restore(
        THRESHOLD.with(|t| t.replace(threshold)),
        COUNT.with(|n| n.replace(0)),
        LIVE.with(|l| l.replace(0)),
        PEAK.with(|p| p.replace(0)),
    );
    let out = f();
    let cost = Cost {
        allocs: COUNT.with(Cell::get),
        peak_bytes: PEAK.with(Cell::get),
    };
    (out, cost)
}

// --- the budgets -------------------------------------------------------------

/// One guarded operation and what it is allowed to cost.
struct Budget {
    /// The `pub fn` in `src/convolution.rs` this row covers.
    op: &'static str,
    /// Which arm of it, where an operation has more than one.
    arm: &'static str,
    run: fn(&Raster) -> Result<Raster, ConvolutionError>,
    /// Image-sized allocations one run is allowed to make.
    allocs: u32,
    /// Image-sized bytes it is allowed to hold live at once, per pixel.
    live_per_pixel: i64,
}

/// Measured on the tree that closed #700, at both sizes below, and set at the
/// measured value rather than above it. None of these carry slack.
///
/// The per-pixel figures read off the algorithms. `try_sharpen`'s 39 is the
/// LabS raster the colourspace conversion returns and the f32 widening of it
/// (twelve bytes a pixel each), the clamped `i32` L plane and the two separable
/// blur passes (four each), and the Rgb8 raster the conversion back produces
/// (three), with the widening and the L plane overlapping the pair that are
/// live longest. Canny's float arm holds two gradient rasters, the two
/// widenings of them and the polar pairs, at twelve or twenty-four bytes a
/// pixel apiece; its uchar arm skips the widenings and reads the gradient bytes
/// directly, which is why it is a third of the float arm.
///
/// Both mutations #700 names add twelve bytes a pixel and one allocation to the
/// sharpen row, one as a `Raster` and one as a `Vec<f32>`.
const BUDGETS: &[Budget] = &[
    Budget {
        op: "try_sharpen",
        arm: "",
        run: |src| src.try_sharpen(1.5, 1.0, 2.0),
        allocs: 6,
        live_per_pixel: 39,
    },
    Budget {
        op: "try_canny",
        arm: "float arm",
        run: |src| src.try_canny(1.4, Precision::Float),
        allocs: 11,
        live_per_pixel: 84,
    },
    Budget {
        op: "try_canny",
        arm: "uchar arm",
        run: |src| src.try_canny(1.4, Precision::Integer),
        allocs: 9,
        live_per_pixel: 33,
    },
];

/// Two sizes, so a budget has to hold as a rate rather than as a constant
/// fitted to one image. Both are far enough above the fixed-size buffers on
/// either path that one byte per pixel cleanly separates image-sized from not.
const DIMS: [u32; 2] = [192, 256];

/// Measure one row at one size, after a warm-up run that keeps any one-time
/// lazily built table out of the window.
fn cost_of(budget: &Budget, dim: u32) -> (Cost, usize) {
    let pixels = (dim as usize) * (dim as usize);
    let src = generate_test_raster(dim, dim).expect("fixture raster");
    (budget.run)(&src).expect("warm-up run");

    let (out, cost) = measure(pixels, || (budget.run)(&src));
    // Not decoration. A cost measured off an early `Err` would sit under every
    // budget here and prove nothing at all, so the run has to have completed
    // and produced a raster of the input's shape.
    let out = out.expect("the operation must run, or a cost under budget says nothing");
    assert_eq!(
        (out.width(), out.height()),
        (dim, dim),
        "{} {} changed the image geometry",
        budget.op,
        budget.arm
    );
    (cost, pixels)
}

#[test]
fn sharpen_and_canny_hold_their_image_sized_allocation_budgets() {
    for budget in BUDGETS {
        for dim in DIMS {
            let (cost, pixels) = cost_of(budget, dim);
            assert!(
                cost.allocs <= budget.allocs,
                "{} {} made {} image-sized allocations at {dim}x{dim} against a budget of {}: \
                 something on the path is copying a whole image again (issue #700)",
                budget.op,
                budget.arm,
                cost.allocs,
                budget.allocs
            );
            let ceiling = budget.live_per_pixel * pixels as i64;
            assert!(
                cost.peak_bytes <= ceiling,
                "{} {} held {} image-sized bytes live at once at {dim}x{dim} ({:.1} a pixel) \
                 against a budget of {} a pixel: something on the path is copying a whole \
                 image again (issue #700)",
                budget.op,
                budget.arm,
                cost.peak_bytes,
                cost.peak_bytes as f64 / pixels as f64,
                budget.live_per_pixel
            );
        }
    }
}

#[test]
fn the_budgets_are_rates_and_not_constants_fitted_to_one_image_size() {
    for budget in BUDGETS {
        let [(small, small_px), (large, large_px)] = DIMS.map(|dim| cost_of(budget, dim));

        assert_eq!(
            small.allocs, large.allocs,
            "{} {} made {} image-sized allocations at {}x{} and {} at {}x{}: the count is \
             supposed to be a property of the algorithm, not of the image",
            budget.op, budget.arm, small.allocs, DIMS[0], DIMS[0], large.allocs, DIMS[1], DIMS[1]
        );
        assert_eq!(
            small.peak_bytes * large_px as i64,
            large.peak_bytes * small_px as i64,
            "{} {} held {:.3} bytes a pixel live at {}x{} and {:.3} at {}x{}: the budget is \
             only meaningful if it is a rate",
            budget.op,
            budget.arm,
            small.peak_bytes as f64 / small_px as f64,
            DIMS[0],
            DIMS[0],
            large.peak_bytes as f64 / large_px as f64,
            DIMS[1],
            DIMS[1]
        );
        // And the rate the row is written at is the rate that was measured, so
        // a budget cannot be quietly padded above what the path actually costs.
        assert_eq!(
            small.peak_bytes,
            budget.live_per_pixel * small_px as i64,
            "{} {}'s recorded budget is not what it measures",
            budget.op,
            budget.arm
        );
        assert_eq!(
            small.allocs, budget.allocs,
            "{} {}'s recorded allocation budget is not what it measures",
            budget.op, budget.arm
        );
    }
}

/// The positive control. Every budget above is an upper bound, and an upper
/// bound is green when the instrument is broken, so the instrument has to be
/// shown moving on a copy that is deliberately put in front of it.
///
/// The copy is the shape of both mutations #700 names: one `clone()` of a
/// buffer already in hand. It has to move the count by exactly one and the
/// peak by exactly what it copied.
#[test]
fn the_allocator_sees_a_deliberate_image_sized_copy() {
    let pixels = 256 * 256;
    let buf = vec![0u8; pixels * 12];

    let (copy, cost) = measure(pixels, || buf.clone());
    assert_eq!(cost.allocs, 1, "one image-sized clone must charge as one");
    assert_eq!(
        cost.peak_bytes,
        (pixels * 12) as i64,
        "and for the size it actually copied"
    );
    drop(copy);

    // A buffer below one byte a pixel is not image-sized and is not charged.
    let small = vec![0u8; pixels - 1];
    let (_, cost) = measure(pixels, || small.clone());
    assert_eq!(cost.allocs, 0, "a sub-image allocation must not be charged");

    // The window closes: nothing is charged once it has returned.
    let (_, cost) = measure(pixels, || ());
    assert_eq!(cost.allocs, 0);
    assert_eq!(cost.peak_bytes, 0);
    let _outside = buf.clone();
    let (_, cost) = measure(pixels, || ());
    assert_eq!(cost.allocs, 0, "the window must not charge its own outside");
}
