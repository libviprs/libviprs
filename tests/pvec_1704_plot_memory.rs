//! Issue #1704: a PDF plot holds one sheet at a time, not the whole document.
//!
//! The peak is measured as live heap bytes with a counting global allocator, the way
//! `pmtiles_bounded_memory.rs` does it, and not as resident size, which never comes back down and so
//! cannot show a peak that stays flat. The drawing is served from memory and makes each sheet's
//! lines on demand, so what it weighs does not change with the sheet count: a peak that grows with
//! the sheets is the plot's own pages. The counter is process wide, so every test takes
//! [`MEASURING`] first and this binary holds nothing else. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

use libviprs::cad::CadSource;
use libviprs::pvec::{SourceUnit, ViewProfile};
use libviprs::pvec_convert::PdfPlot;

#[path = "pvec_1704_support/mod.rs"]
mod support;

static LIVE: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);

struct Counting;

// SAFETY: every call is forwarded to the system allocator unchanged; the counters only observe.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract for `alloc` is passed through as it came.
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            let now =
                LIVE.fetch_add(layout.size() as u64, Ordering::Relaxed) + layout.size() as u64;
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as u64, Ordering::Relaxed);
        // SAFETY: `ptr` and `layout` are the pair the caller got from `alloc`.
        unsafe { System.dealloc(ptr, layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new: usize) -> *mut u8 {
        // SAFETY: the caller's contract for `realloc` is passed through as it came.
        let p = unsafe { System.realloc(ptr, layout, new) };
        if !p.is_null() {
            let (old, new) = (layout.size() as u64, new as u64);
            let now = if new >= old {
                LIVE.fetch_add(new - old, Ordering::Relaxed) + (new - old)
            } else {
                LIVE.fetch_sub(old - new, Ordering::Relaxed) - (old - new)
            };
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        p
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

static MEASURING: Mutex<()> = Mutex::new(());

fn measuring() -> MutexGuard<'static, ()> {
    MEASURING.lock().unwrap_or_else(|e| e.into_inner())
}

const LINES: u32 = 6_000;

/// The most live heap a plot of `sheets` sheets holds above what was live before it started, and
/// the PDF's size.
fn peak_of_plot(sheets: u32) -> (u64, u64) {
    let dir = std::env::temp_dir().join(format!("libviprs-1704m-{sheets}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let dest = dir.join("plot.pdf");
    let bytes = support::drawing(sheets, LINES);
    let before = LIVE.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let out = PdfPlot::new(CadSource::Bytes(&bytes), &dest)
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("S001"))
        .layouts((1..=sheets).map(|n| format!("S{n:03}")))
        .convert_with::<support::Synthetic>()
        .unwrap();
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(before);
    assert_eq!(out.pages, sheets);
    let size = std::fs::metadata(&dest).unwrap().len();
    let _ = std::fs::remove_dir_all(&dir);
    (peak, size)
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1704_the_peak_does_not_move_when_the_sheet_count_grows_fivefold() {
    let _g = measuring();
    let (few, few_size) = peak_of_plot(4);
    let (many, many_size) = peak_of_plot(20);
    assert!(
        many_size > few_size * 3,
        "the larger plot really has more pages"
    );
    // Sixteen more sheets of this size are about ten megabytes of page content. A plot that
    // streams its pages holds one sheet at a time, so what it adds is allocator noise.
    let slack = 3 << 20;
    assert!(
        many <= few + slack,
        "peak {many} bytes for 20 sheets against {few} for 4: grows with the sheets"
    );
}
