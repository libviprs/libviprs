//! Budgets the image-sized allocations of the convolution module: every
//! `pub fn` in `src/convolution.rs` that a row below names, held to the two
//! numbers it measures.
//!
//! It started as the guard for the "no image-sized infallible allocation left
//! on this path" claim that #669 wrote onto `try_sharpen` and `try_canny`
//! (issue #700), and grew `conv`, `sobel`, `gaussblur` and `compass` rows when
//! #575 took the whole-image `f64` widening off the traversal all four of them
//! run through. The file used to be called
//! `sharpen_canny_image_sized_allocations.rs`; the budgets are no longer only
//! sharpen's and canny's, so the name is not either.
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
//! # The instrument checks itself
//!
//! Every budget above is an upper bound, and an upper bound is green when the
//! instrument is broken, so four properties of the instrument are asserted
//! rather than argued for. Each one is here because the argument on its own
//! stayed green under a mutation of the thing it claimed:
//!
//! * **All four allocator arms are exercised.** `Vec::clone` reaches `alloc`,
//!   `vec![0u8; n]` reaches `alloc_zeroed`, a `Vec` grown by resizing reaches
//!   `realloc`, and dropping any of them reaches `dealloc`. Breaking the
//!   `realloc` or `alloc_zeroed` arm used to leave the whole file green,
//!   because nothing on the sharpen or canny path takes either route today, so
//!   a new `vec![0; n]` buffer would have arrived on a guard that had silently
//!   stopped working.
//! * **The counter is thread-local, so the rows say so out loud.** Every row
//!   also reads a process-global counter and requires the two to agree. An
//!   image-sized allocation moved onto a worker thread is `0` to the
//!   thread-local counter and `1` to the global one, which the positive
//!   control demonstrates, so the rows would catch a path that quietly fanned
//!   out instead of measuring it as free.
//! * **`LIVE` never goes negative on a guarded row.** It is a signed counter
//!   for two reasons, and both are asymmetries rather than bugs: a window can
//!   outlive a buffer allocated before it opened, and `release` credits
//!   against the threshold in force at free time rather than the one in force
//!   when the buffer was charged (the two [`DIMS`] use different thresholds).
//!   Neither is reachable from a guarded row today, and the low-water mark
//!   assertion is what says so.
//! * **`charge` never re-enters itself.** The thread-locals are `Cell`s of
//!   `Copy` scalars with `const` initialisers and no destructor, so on a target
//!   where std gives them the `#[thread_local]` fast path, touching one
//!   allocates nothing. Measured depth is 1 on `aarch64-apple-darwin` and on
//!   `x86_64-unknown-linux-gnu`, which are the two targets this ships on. On a
//!   target where std falls back to `pthread_key`, the first access on a thread
//!   boxes its value, and boxing from inside `alloc` recurses until the stack
//!   dies; the depth assertion is what would name that rather than leaving a
//!   crash to be diagnosed.
//!
//! # What this cannot see
//!
//! * It cannot tell an image-sized allocation that went through a fallible
//!   reservation from one that did not. It is a budget, not a proof of
//!   fallibility. What it proves is that no *further* image-sized buffer fits,
//!   which is the property both mutations break. The funnel count in
//!   `src/raster.rs` is the half that answers the other question, and the
//!   section below says why the two are in different binaries.
//! * It counts a request, not a residency. An allocator that over-allocates, or
//!   a page never touched, is charged at the size asked for.
//! * The budgets only cover the carriers the rows name. The sharpen, canny and
//!   `conv` rows run an `Rgb8` input and an `Rgba8` one, so a buffer that only
//!   appears with an alpha band is visible; `conv` adds a 16-bit and a
//!   32-bit-float carrier and `sobel` a single-band one, which between them
//!   reach all three arms of the row widening and all three band counts, but no
//!   row here covers a 16-bit or float carrier through sharpen or canny.
//! * A change that legitimately adds an image-sized buffer to one of these
//!   paths reddens the row it belongs to, and the budget wants re-measuring
//!   with the same evidence that set it in the first place. That is the
//!   intended cost.
//!
//! # One instrument, two halves, and why they are in different binaries
//!
//! `#[global_allocator]` here is scoped to this one integration-test binary, so
//! it reaches no other test and it is not a process-wide install. #696 wanted a
//! counting allocator too, to prove that every image-sized allocation on a path
//! went through the fallible reservation helper, and the crate having two of
//! these with different accounting is how a third gets invented. So it did not
//! build one. It counts consumptions of the `cfg(test)` probe on
//! `raster::try_plane` instead, per entry point and per module, at exact
//! equality, in
//! `raster::tests::every_plane_these_paths_reserve_goes_through_the_one_funnel`.
//!
//! That check cannot live here and this one cannot live there. The probe is
//! `cfg(test)`, so it exists only in the library's own unit-test binary, and an
//! integration test links the library built *without* `cfg(test)`;
//! `#[global_allocator]` is scoped to the binary that installs it, and the unit
//! test binary is not this one. So the two halves sit in the two places each is
//! measurable, and they are read together: where a row there and a row here
//! agree for the same operation, every image-sized allocation on that path went
//! through the fallible helper. `try_sharpen` is six in both. `try_conv` is one
//! here and two there, because the row window is a real reservation that is not
//! image-sized, which is the property #575 put there on purpose.
//!
//! Neither half is worth much alone, and each covers the other's blind spot:
//! this one cannot tell a fallible reservation from an infallible one, and that
//! one cannot see a buffer that never reaches the helper at all. That decision
//! and the rule behind it are written up in `CONTRIBUTING.md`, under
//! "Allocation instruments: one shape, two questions", because the next person
//! to want one will read that and may never open this file.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};

use libviprs::{
    Angle45, Combine, ConvolutionError, Kernel, PixelFormat, Precision, Raster,
    generate_test_raster,
};

// --- counting global allocator ----------------------------------------------

thread_local! {
    /// Size at or above which an allocation is charged, in bytes. `usize::MAX`
    /// disarms, which is where every thread starts and where the measurement
    /// window leaves it, so tests running in parallel never see one another.
    static THRESHOLD: Cell<usize> = const { Cell::new(usize::MAX) };
    /// How many charged allocations this thread has made in the window.
    static COUNT: Cell<u32> = const { Cell::new(0) };
    /// Charged bytes currently live. Signed, because two asymmetries can drive
    /// it below zero: the window can outlive a buffer allocated before it
    /// opened, and `release` credits against the threshold in force at free
    /// time rather than the one in force when the buffer was charged. Neither
    /// is reachable from a guarded row, and [`Cost::min_live`] is what holds
    /// that rather than the comment.
    static LIVE: Cell<i64> = const { Cell::new(0) };
    /// The high-water mark `LIVE` reached in the window.
    static PEAK: Cell<i64> = const { Cell::new(0) };
    /// The low-water mark `LIVE` reached in the window.
    static TROUGH: Cell<i64> = const { Cell::new(0) };
    /// How deep inside [`charge`] this thread is right now.
    static DEPTH: Cell<u32> = const { Cell::new(0) };
    /// The deepest [`DEPTH`] reached in the window. Anything above 1 means a
    /// thread-local access allocated, which on a `pthread_key` target is the
    /// first step of an unbounded recursion inside `alloc`.
    static MAX_DEPTH: Cell<u32> = const { Cell::new(0) };
}

/// Threshold for the process-global counter, armed only inside a window.
static GLOBAL_THRESHOLD: AtomicUsize = AtomicUsize::new(usize::MAX);
/// Charged allocations made anywhere in the process during the window, on any
/// thread. Compared against [`COUNT`] so a path that fanned out onto a worker
/// thread cannot go on measuring as if it had not.
static GLOBAL_COUNT: AtomicU32 = AtomicU32::new(0);

/// Every test here arms the process-global counter and allocates image-sized
/// buffers, so they run one at a time instead of counting one another.
static SERIALISE: Mutex<()> = Mutex::new(());

/// Take the measuring lock, ignoring poisoning: a failure in one test should
/// show up as one red test rather than as three.
fn serialised() -> MutexGuard<'static, ()> {
    SERIALISE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Charge an allocation.
///
/// Every cell is a `Cell` of a `Copy` scalar with a `const` initialiser and no
/// destructor, so on the `#[thread_local]` fast path touching one allocates
/// nothing and cannot recurse back into this allocator. [`MAX_DEPTH`] is what
/// checks that rather than assuming it. `try_with` covers the one case left, a
/// thread whose thread-local storage is already being torn down, by declining
/// to charge rather than panicking inside the allocator, and the arithmetic
/// saturates so that no input can unwind out of `GlobalAlloc::alloc` either.
fn charge(size: usize) {
    let _ = DEPTH.try_with(|d| {
        let depth = d.get().saturating_add(1);
        d.set(depth);
        let _ = MAX_DEPTH.try_with(|m| {
            if depth > m.get() {
                m.set(depth);
            }
        });
        charge_inner(size);
        d.set(d.get().saturating_sub(1));
    });
}

/// The charging itself, wrapped by [`charge`]'s depth accounting. Split out so
/// that the depth is incremented before anything else is touched: the hazard
/// being watched for is a thread-local access allocating, and the watch has to
/// be armed before the first such access rather than after it.
fn charge_inner(size: usize) {
    if size >= GLOBAL_THRESHOLD.load(Ordering::Relaxed) {
        GLOBAL_COUNT.fetch_add(1, Ordering::Relaxed);
    }
    let _ = THRESHOLD.try_with(|t| {
        if size >= t.get() {
            let bytes = i64::try_from(size).unwrap_or(i64::MAX);
            let _ = COUNT.try_with(|n| n.set(n.get().wrapping_add(1)));
            let _ = LIVE.try_with(|l| {
                let now = l.get().saturating_add(bytes);
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
/// allocation and still counts as one. It carries no depth accounting either,
/// because the recursion this file is watching for starts inside `alloc` and
/// [`MAX_DEPTH`] is where it would show up.
fn release(size: usize) {
    let _ = THRESHOLD.try_with(|t| {
        if size >= t.get() {
            let bytes = i64::try_from(size).unwrap_or(i64::MAX);
            let _ = LIVE.try_with(|l| {
                let now = l.get().saturating_sub(bytes);
                l.set(now);
                let _ = TROUGH.try_with(|m| {
                    if now < m.get() {
                        m.set(now);
                    }
                });
            });
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
/// live image-sized bytes, plus the three self-checks on the instrument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Cost {
    allocs: u32,
    peak_bytes: i64,
    /// The same count taken process-wide instead of thread-locally.
    global_allocs: u32,
    /// The low-water mark of live charged bytes.
    min_live: i64,
    /// The deepest re-entry into [`charge`].
    max_depth: u32,
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
    struct Restore {
        threshold: usize,
        count: u32,
        live: i64,
        peak: i64,
        trough: i64,
        max_depth: u32,
        global_threshold: usize,
        global_count: u32,
    }
    impl Drop for Restore {
        fn drop(&mut self) {
            // Both thresholds first and adjacent, so the two counters cannot
            // disagree about where the window ends, and then `try_with` for the
            // same reason every other access in this file uses it: this runs on
            // the unwind path too, and a panic escaping here would be a panic
            // escaping a `Drop` during unwinding.
            GLOBAL_THRESHOLD.store(self.global_threshold, Ordering::Relaxed);
            let _ = THRESHOLD.try_with(|t| t.set(self.threshold));
            GLOBAL_COUNT.store(self.global_count, Ordering::Relaxed);
            let _ = COUNT.try_with(|n| n.set(self.count));
            let _ = LIVE.try_with(|l| l.set(self.live));
            let _ = PEAK.try_with(|p| p.set(self.peak));
            let _ = TROUGH.try_with(|m| m.set(self.trough));
            let _ = MAX_DEPTH.try_with(|m| m.set(self.max_depth));
        }
    }
    // Field initialisers run in the order written, so every counter is zeroed
    // before either threshold is armed and the two thresholds go up adjacently.
    // Nothing in here allocates, so the window has the same edge for both.
    let _restore = Restore {
        count: COUNT.with(|n| n.replace(0)),
        live: LIVE.with(|l| l.replace(0)),
        peak: PEAK.with(|p| p.replace(0)),
        trough: TROUGH.with(|m| m.replace(0)),
        max_depth: MAX_DEPTH.with(|m| m.replace(0)),
        global_count: GLOBAL_COUNT.swap(0, Ordering::Relaxed),
        threshold: THRESHOLD.with(|t| t.replace(threshold)),
        global_threshold: GLOBAL_THRESHOLD.swap(threshold, Ordering::Relaxed),
    };
    let out = f();
    let cost = Cost {
        allocs: COUNT.with(Cell::get),
        peak_bytes: PEAK.with(Cell::get),
        global_allocs: GLOBAL_COUNT.load(Ordering::Relaxed),
        min_live: TROUGH.with(Cell::get),
        max_depth: MAX_DEPTH.with(Cell::get),
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
    /// The carrier the row runs on. A budget only covers the one it measured.
    carrier: &'static str,
    src: fn(u32) -> Raster,
    run: fn(&Raster) -> Result<Raster, ConvolutionError>,
    /// Image-sized allocations one run is allowed to make.
    allocs: u32,
    /// Image-sized bytes it is allowed to hold live at once, per pixel.
    live_per_pixel: i64,
}

/// The 3-band 8-bit fixture, which is what `generate_test_raster` produces.
fn rgb8(dim: u32) -> Raster {
    generate_test_raster(dim, dim).expect("fixture raster")
}

/// The same gradient with an alpha band on it.
///
/// Without this every row runs a 3-band input and an image-sized buffer that
/// only appears when there is an alpha band to carry is invisible to all of
/// them. The alpha ramp is deliberately not constant, so an implementation that
/// short-circuits an opaque alpha does not accidentally get measured as if the
/// band were not there.
fn rgba8(dim: u32) -> Raster {
    let rgb = rgb8(dim);
    let px = (dim as usize) * (dim as usize);
    let mut data = vec![0u8; px * 4];
    for (p, out) in data.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        out[..3].copy_from_slice(&rgb.data()[p * 3..p * 3 + 3]);
        out[3] = (p % 256) as u8;
    }
    Raster::new(dim, dim, PixelFormat::Rgba8, data).expect("fixture raster")
}

/// One band of the gradient, so a buffer that scales with the band count and
/// one that does not are told apart by something other than the alpha row.
fn gray8(dim: u32) -> Raster {
    let rgb = rgb8(dim);
    let px = (dim as usize) * (dim as usize);
    let data: Vec<u8> = (0..px).map(|p| rgb.data()[p * 3]).collect();
    Raster::new(dim, dim, PixelFormat::Gray8, data).expect("fixture raster")
}

/// The gradient on the 16-bit carrier.
///
/// `Scan::advance` widens three carriers with three different decodes, and
/// until this row existed nothing here ran the 2-byte one: every guarded path
/// was 8-bit, or 32-bit float through canny's gradient stage. An arm no row
/// exercises is an arm that can break with the whole file green, which is the
/// same argument the positive control makes about the allocator's four.
fn rgb16(dim: u32) -> Raster {
    let rgb = rgb8(dim);
    let n = (dim as usize) * (dim as usize) * 3;
    let mut data = vec![0u8; n * 2];
    for (i, out) in data.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        *out = (u16::from(rgb.data()[i]) * 257).to_ne_bytes();
    }
    Raster::new(dim, dim, PixelFormat::Rgb16, data).expect("fixture raster")
}

/// The gradient on the 32-bit float carrier, which is the third decode and
/// also the only carrier that reaches `vips_convi_gen`'s double path.
fn rgbf32(dim: u32) -> Raster {
    let rgb = rgb8(dim);
    let n = (dim as usize) * (dim as usize) * 3;
    let mut data = vec![0u8; n * 4];
    for (i, out) in data.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        *out = (f32::from(rgb.data()[i]) / 255.0).to_ne_bytes();
    }
    let fmt = PixelFormat::with_channels(3, 4).expect("three bands have a float format");
    Raster::new(dim, dim, fmt, data).expect("fixture raster")
}

/// The template the two correlation rows run against.
///
/// Small on purpose: the template is the operand read whole at every output
/// sample, and both operations are `O(w * h * tw * th)`, so a large one would
/// cost the suite minutes and pin nothing a small one does not.
fn correlation_template() -> Raster {
    generate_test_raster(3, 3).expect("fixture template")
}

/// The 3x3 box blur every `conv` and `compass` row runs, which is the mask
/// shape the module's own pins use.
fn box3() -> Kernel {
    Kernel {
        data: vec![vec![1.0; 3]; 3],
        scale: 9.0,
    }
}

/// Measured at both sizes below and set at the measured value rather than
/// above it. None of these carry slack.
///
/// The per-pixel figures read off the algorithms. `try_sharpen`'s 39 on `Rgb8`
/// is the LabS raster the colourspace conversion returns and the f32 widening
/// of it (twelve bytes a pixel each), the clamped `i32` L plane and the two
/// separable blur passes (four each), and the Rgb8 raster the conversion back
/// produces (three), with the widening and the L plane overlapping the pair
/// that are live longest. Canny's float arm holds two gradient rasters, the two
/// widenings of them and the polar pairs, at twelve or twenty-four bytes a
/// pixel apiece; its uchar arm skips the widenings and reads the gradient bytes
/// directly, which is why it is a fifth of the float arm.
///
/// The `Rgba8` rows are the same algorithms carrying a fourth band, and their
/// numbers say so: the allocation counts are identical and both canny rates are
/// exactly four thirds of the `Rgb8` ones (84 to 112, 15 to 20). Sharpen is 39
/// to 48 rather than 52 because only the buffers that carry every band widen
/// with the alpha (the LabS raster and its widening go twelve to sixteen, the
/// output raster three to four) while the L plane and the two blur passes are
/// single-band and do not move. A buffer that only appeared with an alpha band
/// would sit outside all six of those relationships.
///
/// Both mutations #700 names add twelve bytes a pixel and one allocation to the
/// sharpen row, one as a `Raster` and one as a `Vec<f32>`.
///
/// # The traversal rows
///
/// `conv`, `sobel`, `gaussblur` and `compass` all run `Scan`, and their rows
/// are what #575 moved. Each one used to carry a whole-image `f64` widening,
/// eight bytes a sample, on top of what it reads here now:
///
/// | row | before #575 | after |
/// |---|---|---|
/// | `try_conv` integer, `Rgb8` | 2 allocs, 27 a pixel | 1, 3 |
/// | `try_conv` float, `Rgb8` | 2, 36 | 1, 12 |
/// | `try_sobel`, `Rgb8` | 2, 27 | 1, 3 |
/// | `try_gaussblur` integer, `Rgb8` | 4, 30 | 2, 6 |
/// | `try_canny` uchar, `Rgb8` | 9, 33 | 6, 15 |
///
/// So a `conv` at integer precision now allocates **one** image-sized buffer,
/// its own output, and holds nothing else: three bytes a pixel for a three-band
/// uchar image is the output and nothing more. The window itself is real but it
/// is not image-sized, which is exactly the property these rows pin: it is
/// `span * width * bands * 8` bytes for a mask `span` rows tall, so at 192 and
/// 256 square it sits below one byte a pixel and is not charged at either, and
/// a window that went back to scaling with the height would be.
///
/// The float arm reads twelve rather than three because `vips_convf` writes a
/// float image whatever it was handed, so the output is four bytes a band; the
/// `Rgb16` row reads six for the same reason in reverse, since `vips_convi`
/// keeps the input depth.
///
/// `try_compass` holds its `times` results live and nothing else: at
/// `times = 4` that is four uchar rasters, the `f64` accumulator and the
/// output, 39 bytes a pixel. It used to be 159, because it widened every one
/// of those results to `f64` to combine them and held all four widenings at
/// once, which was 96 of the 159 and made compass the most expensive operation
/// in the crate at 36 times its input (issue #790). Holding the results
/// themselves is inherent: `vips_compass` convolves `times` times and combines
/// the absolute results, so `times * bands` bytes a pixel is the floor.
///
/// The two correlation rows are one allocation and 12 bytes a pixel, which is
/// the float output raster and nothing else. They were 60: the image widened
/// to `f64` (24), a `Vec<f64>` of results written and read in output order
/// (24) and the raster built from it (12). The widening came off when they
/// went onto the same `RowWindow` the traversal uses, and the result buffer
/// when they started writing into the output raster directly (issue #791).
/// Their template is still widened whole and is deliberately 3x3, so it sits
/// under the threshold: it is bounded by the operand a caller passes rather
/// than by the image.
const BUDGETS: &[Budget] = &[
    Budget {
        op: "try_conv",
        arm: "integer arm",
        carrier: "Rgb8",
        src: rgb8,
        run: |src| src.try_conv(&box3(), Precision::Integer),
        allocs: 1,
        live_per_pixel: 3,
    },
    Budget {
        op: "try_conv",
        arm: "float arm",
        carrier: "Rgb8",
        src: rgb8,
        run: |src| src.try_conv(&box3(), Precision::Float),
        allocs: 1,
        live_per_pixel: 12,
    },
    Budget {
        op: "try_conv",
        arm: "integer arm",
        carrier: "Rgba8",
        src: rgba8,
        run: |src| src.try_conv(&box3(), Precision::Integer),
        allocs: 1,
        live_per_pixel: 4,
    },
    Budget {
        op: "try_conv",
        arm: "float arm",
        carrier: "Rgba8",
        src: rgba8,
        run: |src| src.try_conv(&box3(), Precision::Float),
        allocs: 1,
        live_per_pixel: 16,
    },
    Budget {
        op: "try_conv",
        arm: "integer arm",
        carrier: "Rgb16",
        src: rgb16,
        run: |src| src.try_conv(&box3(), Precision::Integer),
        allocs: 1,
        live_per_pixel: 6,
    },
    Budget {
        op: "try_conv",
        arm: "float arm",
        carrier: "FloatF32(3)",
        src: rgbf32,
        run: |src| src.try_conv(&box3(), Precision::Float),
        allocs: 1,
        live_per_pixel: 12,
    },
    Budget {
        op: "try_sobel",
        arm: "",
        carrier: "Rgb8",
        src: rgb8,
        run: Raster::try_sobel,
        allocs: 1,
        live_per_pixel: 3,
    },
    Budget {
        op: "try_sobel",
        arm: "",
        carrier: "Gray8",
        src: gray8,
        run: Raster::try_sobel,
        allocs: 1,
        live_per_pixel: 1,
    },
    Budget {
        op: "try_gaussblur",
        arm: "integer arm",
        carrier: "Rgb8",
        src: rgb8,
        run: |src| src.try_gaussblur(1.4, 0.2, Precision::Integer),
        allocs: 2,
        live_per_pixel: 6,
    },
    Budget {
        op: "try_compass",
        arm: "Max, 4 rounds",
        carrier: "Rgb8",
        src: rgb8,
        run: |src| src.try_compass(&box3(), 4, Angle45::D45, Combine::Max, Precision::Integer),
        allocs: 6,
        live_per_pixel: 39,
    },
    Budget {
        op: "try_spcor",
        arm: "3x3 template",
        carrier: "Rgb8",
        src: rgb8,
        run: |src| src.try_spcor(&correlation_template()),
        allocs: 1,
        live_per_pixel: 12,
    },
    Budget {
        op: "try_fastcor",
        arm: "3x3 template",
        carrier: "Rgb8",
        src: rgb8,
        run: |src| src.try_fastcor(&correlation_template()),
        allocs: 1,
        live_per_pixel: 12,
    },
    Budget {
        op: "try_sharpen",
        arm: "",
        carrier: "Rgb8",
        src: rgb8,
        run: |src| src.try_sharpen(1.5, 1.0, 2.0),
        allocs: 6,
        live_per_pixel: 39,
    },
    Budget {
        op: "try_canny",
        arm: "float arm",
        carrier: "Rgb8",
        src: rgb8,
        run: |src| src.try_canny(1.4, Precision::Float),
        allocs: 8,
        live_per_pixel: 84,
    },
    Budget {
        op: "try_canny",
        arm: "uchar arm",
        carrier: "Rgb8",
        src: rgb8,
        run: |src| src.try_canny(1.4, Precision::Integer),
        allocs: 6,
        live_per_pixel: 15,
    },
    Budget {
        op: "try_sharpen",
        arm: "",
        carrier: "Rgba8",
        src: rgba8,
        run: |src| src.try_sharpen(1.5, 1.0, 2.0),
        allocs: 6,
        live_per_pixel: 48,
    },
    Budget {
        op: "try_canny",
        arm: "float arm",
        carrier: "Rgba8",
        src: rgba8,
        run: |src| src.try_canny(1.4, Precision::Float),
        allocs: 8,
        live_per_pixel: 112,
    },
    Budget {
        op: "try_canny",
        arm: "uchar arm",
        carrier: "Rgba8",
        src: rgba8,
        run: |src| src.try_canny(1.4, Precision::Integer),
        allocs: 6,
        live_per_pixel: 20,
    },
];

/// Two sizes, so a budget has to hold as a rate rather than as a constant
/// fitted to one image. Both are far enough above the fixed-size buffers on
/// these paths that one byte per pixel cleanly separates image-sized from not,
/// and that now includes the convolution row window: it is
/// `span * width * bands * 8` bytes, which for the mask shapes here is under
/// `dim * dim` at both sizes. A window that went back to scaling with the
/// height would cross the threshold and be charged, which is the point.
const DIMS: [u32; 2] = [192, 256];

/// Measure one row at one size, after a warm-up run that keeps any one-time
/// lazily built table out of the window.
fn cost_of(budget: &Budget, dim: u32) -> (Cost, usize) {
    let pixels = (dim as usize) * (dim as usize);
    let src = (budget.src)(dim);
    (budget.run)(&src).expect("warm-up run");

    let (out, cost) = measure(pixels, || (budget.run)(&src));
    // Not decoration. A cost measured off an early `Err` would sit under every
    // budget here and prove nothing at all, so the run has to have completed
    // and produced a raster of the input's shape.
    let out = out.expect("the operation must run, or a cost under budget says nothing");
    assert_eq!(
        (out.width(), out.height()),
        (dim, dim),
        "{} {} on {} changed the image geometry",
        budget.op,
        budget.arm,
        budget.carrier
    );
    assert_eq!(
        cost.global_allocs, cost.allocs,
        "{} {} on {} charged {} image-sized allocations to the thread running it and {} to the \
         whole process at {dim}x{dim}: the budgets are counted thread-locally, so a path that \
         fans out stops being measured (issue #700)",
        budget.op, budget.arm, budget.carrier, cost.allocs, cost.global_allocs
    );
    assert!(
        cost.min_live >= 0,
        "{} {} on {} drove live charged bytes down to {} at {dim}x{dim}: a buffer charged \
         outside the window was credited inside it, so the peak is measured against the wrong \
         floor",
        budget.op,
        budget.arm,
        budget.carrier,
        cost.min_live
    );
    assert_eq!(
        cost.max_depth, 1,
        "{} {} on {} re-entered the counting allocator {} deep at {dim}x{dim}: a thread-local \
         access is allocating, which on this target it must not",
        budget.op, budget.arm, budget.carrier, cost.max_depth
    );
    (cost, pixels)
}

#[test]
fn every_convolution_op_holds_its_image_sized_allocation_budget() {
    let _serial = serialised();
    for budget in BUDGETS {
        for dim in DIMS {
            let (cost, pixels) = cost_of(budget, dim);
            assert!(
                cost.allocs <= budget.allocs,
                "{} {} on {} made {} image-sized allocations at {dim}x{dim} against a budget of \
                 {}: something on the path is copying a whole image again (issues #700, #575)",
                budget.op,
                budget.arm,
                budget.carrier,
                cost.allocs,
                budget.allocs
            );
            let ceiling = budget.live_per_pixel * pixels as i64;
            assert!(
                cost.peak_bytes <= ceiling,
                "{} {} on {} held {} image-sized bytes live at once at {dim}x{dim} ({:.1} a \
                 pixel) against a budget of {} a pixel: something on the path is copying a whole \
                 image again (issues #700, #575)",
                budget.op,
                budget.arm,
                budget.carrier,
                cost.peak_bytes,
                cost.peak_bytes as f64 / pixels as f64,
                budget.live_per_pixel
            );
        }
    }
}

#[test]
fn the_budgets_are_rates_and_not_constants_fitted_to_one_image_size() {
    let _serial = serialised();
    for budget in BUDGETS {
        let [(small, small_px), (large, large_px)] = DIMS.map(|dim| cost_of(budget, dim));

        assert_eq!(
            small.allocs,
            large.allocs,
            "{} {} on {} made {} image-sized allocations at {}x{} and {} at {}x{}: the count is \
             supposed to be a property of the algorithm, not of the image",
            budget.op,
            budget.arm,
            budget.carrier,
            small.allocs,
            DIMS[0],
            DIMS[0],
            large.allocs,
            DIMS[1],
            DIMS[1]
        );
        assert_eq!(
            small.peak_bytes * large_px as i64,
            large.peak_bytes * small_px as i64,
            "{} {} on {} held {:.3} bytes a pixel live at {}x{} and {:.3} at {}x{}: the budget is \
             only meaningful if it is a rate",
            budget.op,
            budget.arm,
            budget.carrier,
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
            "{} {} on {}'s recorded budget is not what it measures",
            budget.op,
            budget.arm,
            budget.carrier
        );
        assert_eq!(
            small.allocs, budget.allocs,
            "{} {} on {}'s recorded allocation budget is not what it measures",
            budget.op, budget.arm, budget.carrier
        );
    }
}

/// The positive control. Every budget above is an upper bound, and an upper
/// bound is green when the instrument is broken, so the instrument has to be
/// shown moving on copies that are deliberately put in front of it.
///
/// One `clone()` of a buffer already in hand is the shape of both mutations
/// #700 names, and it has to move the count by exactly one and the peak by
/// exactly what it copied. The rest of this covers the arms of the allocator
/// and the properties of the counter that nothing on the guarded paths happens
/// to exercise today, because an arm nothing exercises is an arm that can break
/// with the whole file green.
#[test]
fn the_allocator_sees_a_deliberate_image_sized_copy() {
    let _serial = serialised();
    let pixels = 256 * 256;
    let buf = vec![0u8; pixels * 12];

    // `Vec::clone` reaches `alloc`.
    let (copy, cost) = measure(pixels, || buf.clone());
    assert_eq!(cost.allocs, 1, "one image-sized clone must charge as one");
    assert_eq!(
        cost.peak_bytes,
        (pixels * 12) as i64,
        "and for the size it actually copied"
    );
    assert_eq!(cost.global_allocs, 1, "and the global counter must see it");
    assert_eq!(
        cost.min_live, 0,
        "nothing was freed, so nothing was credited"
    );
    assert_eq!(
        cost.max_depth, 1,
        "charging must not re-enter the allocator: a thread-local access allocated"
    );
    drop(copy);

    // `vec![0u8; n]` reaches `alloc_zeroed`, which is a different arm and is
    // not on either guarded path today. Breaking it left this file green.
    let (zeroed, cost) = measure(pixels, || vec![0u8; pixels * 12]);
    assert_eq!(
        cost.allocs, 1,
        "a zeroed image-sized buffer must charge as one"
    );
    assert_eq!(
        cost.peak_bytes,
        (pixels * 12) as i64,
        "and for the size it asked for"
    );
    drop(zeroed);

    // A `Vec` grown into image size reaches `realloc`, the fourth arm and the
    // other one nothing on the guarded paths exercises. It has to charge each
    // step, and the peak has to land on the final size rather than on the sum
    // of the steps, which is what crediting the old size is for.
    let mut grown: Vec<u8> = Vec::with_capacity(1);
    let (_, cost) = measure(pixels, || {
        grown.resize(pixels, 0);
        grown.resize(pixels * 2, 0);
        grown.resize(pixels * 4, 0);
    });
    assert_eq!(
        cost.allocs, 3,
        "each growth step at or above the threshold is its own charge"
    );
    assert_eq!(
        cost.peak_bytes,
        (pixels * 4) as i64,
        "a grown buffer peaks at its final size, not at the sum of its steps"
    );
    drop(grown);

    // The counter is thread-local, and this is what that costs: an image-sized
    // allocation on a worker thread is invisible to it. The guarded rows assert
    // the two counters agree, and this is the demonstration that they can
    // disagree, without which that assertion would be asserting nothing.
    let (_, cost) = measure(pixels, || {
        std::thread::spawn(move || {
            let off_thread = vec![0u8; pixels * 12];
            std::hint::black_box(&off_thread);
        })
        .join()
        .expect("worker thread");
    });
    assert_eq!(
        cost.allocs, 0,
        "the thread-local counter cannot see a worker thread, which is the point"
    );
    assert_eq!(
        cost.global_allocs, 1,
        "and the process-global one has to, or the fan-out check on each row is vacuous"
    );

    // `LIVE` is signed because a window can outlive a buffer allocated before
    // it opened. That is the documented asymmetry, and here it is, driving the
    // low-water mark negative. No guarded row may do this.
    let early = vec![0u8; pixels * 12];
    let (_, cost) = measure(pixels, || drop(early));
    assert_eq!(cost.allocs, 0, "a free is not an allocation");
    assert_eq!(
        cost.min_live,
        -((pixels * 12) as i64),
        "crediting a buffer the window never charged has to show up as a negative floor"
    );

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
    assert_eq!(
        cost.global_allocs, 0,
        "and neither must the global counter, which is armed by the window too"
    );
}

/// [`charge`] runs inside `GlobalAlloc::alloc`, where unwinding is not allowed.
/// Its arithmetic is unreachable at the limits (2^32 charged allocations in one
/// window, 9.2 exabytes live), but "unreachable" is an argument and this is a
/// check, and saturating is free.
#[test]
fn the_counters_saturate_instead_of_unwinding_out_of_the_allocator() {
    let _serial = serialised();
    let (_, cost) = measure(4096, || {
        COUNT.with(|n| n.set(u32::MAX));
        LIVE.with(|l| l.set(i64::MAX));
        charge(4096);
        LIVE.with(|l| l.set(i64::MIN));
        release(4096);
    });
    assert_eq!(cost.allocs, 0, "the count wraps rather than panicking");
    assert_eq!(cost.peak_bytes, i64::MAX, "live bytes saturate at the top");
    assert_eq!(cost.min_live, i64::MIN, "and at the bottom");
}

/// English numerals, so the count `CONTRIBUTING.md` spells out can be compared
/// against the one this file measures. Wide enough that the table would have to
/// grow a long way before this stopped answering, and the check says so out
/// loud rather than passing when it falls off the end.
const NUMERALS: [&str; 41] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "twenty-one",
    "twenty-two",
    "twenty-three",
    "twenty-four",
    "twenty-five",
    "twenty-six",
    "twenty-seven",
    "twenty-eight",
    "twenty-nine",
    "thirty",
    "thirty-one",
    "thirty-two",
    "thirty-three",
    "thirty-four",
    "thirty-five",
    "thirty-six",
    "thirty-seven",
    "thirty-eight",
    "thirty-nine",
    "forty",
];

/// `CONTRIBUTING.md` tells the next lane what touching `src/convolution.rs`
/// costs, and the number it quotes has to be the number of rows in [`BUDGETS`].
///
/// It was not. #575 put four operations' worth of rows in this file and took
/// the table from sixteen to eighteen, and the paragraph that warns about the
/// cost went on saying sixteen, so the one document a lane reads *before*
/// starting understated the blast radius by two rows. That is the shape this
/// repo keeps finding: a doc claim with nothing holding it. Now something
/// holds it.
///
/// `include_str!` rather than a read, deliberately: the file is baked in at
/// compile time, so this touches no filesystem at run time and stays off the
/// Miri inventory.
#[test]
fn contributing_quotes_the_number_of_rows_this_file_has() {
    const CONTRIBUTING: &str = include_str!("../CONTRIBUTING.md");
    const PHRASE: &str = " rows of two numbers";

    let occurrences = CONTRIBUTING.matches(PHRASE).count();
    assert_eq!(
        occurrences, 1,
        "CONTRIBUTING.md says \"{PHRASE}\" {occurrences} times; this check reads the one \
         occurrence it expects, so a second one has to be reconciled by hand"
    );

    let quoted = CONTRIBUTING
        .split_once(PHRASE)
        .map(|(before, _)| before)
        .and_then(|before| before.rsplit(['*', ' ']).find(|w| !w.is_empty()))
        .expect("the phrase is there, so a word comes before it");
    let want = NUMERALS
        .get(BUDGETS.len())
        .unwrap_or_else(|| panic!("{} rows is past the numeral table", BUDGETS.len()));
    assert_eq!(
        quoted,
        *want,
        "CONTRIBUTING.md tells a lane the convolution budgets are \"{quoted}{PHRASE}\" and this \
         file holds {} of them; a row added or removed here has to move that sentence too",
        BUDGETS.len()
    );
}
