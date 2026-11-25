//! Issue #1713: a plot takes only the model leaves a viewport shows, and refuses by typed code when
//! a stage would hold more than `max_allocation_bytes`.
//!
//! The peak is live heap bytes from a counting global allocator, as in `pvec_1704_plot_memory.rs`; the
//! counter is process wide, so every test takes [`MEASURING`]. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::Duration;

use libviprs::CancelToken;
use libviprs::cad::CadSource;
use libviprs::pvec::{SourceLimits, SourceUnit, ViewProfile};
use libviprs::pvec_convert::{PdfPlot, PvecLayoutSet};

#[path = "pvec_1713_support/mod.rs"]
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

const SHEETS: u32 = 3;
const MODEL: u32 = 60_000;
/// A window of 28 model units tall: 40 by 28 of a model 1000 wide and 60 tall.
const SMALL: u32 = 28;
/// A window that shows the whole model.
const WHOLE: u32 = 700;

fn dir_for(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("libviprs-1713-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn plot<'a>(bytes: &'a [u8], dest: &'a Path, limits: Option<SourceLimits>) -> PdfPlot<'a> {
    let p = PdfPlot::new(CadSource::Bytes(bytes), dest)
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("S001"))
        .layouts((1..=SHEETS).map(|n| format!("S{n:03}")));
    match limits {
        Some(l) => p.limits(l),
        None => p,
    }
}

/// The most live heap a plot holds above what was live before it started.
fn peak_of(bytes: &[u8], name: &str) -> u64 {
    let dir = dir_for(name);
    let dest = dir.join("plot.pdf");
    let before = LIVE.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    plot(bytes, &dest, None)
        .convert_with::<support::Synthetic>()
        .unwrap();
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(before);
    let _ = std::fs::remove_dir_all(&dir);
    peak
}

fn limits(bytes: u64) -> SourceLimits {
    let mut l = SourceLimits::default();
    l.max_allocation_bytes = bytes;
    l
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1713_a_small_window_holds_less_than_the_whole_model() {
    let _g = measuring();
    let small = peak_of(&support::drawing(SHEETS, MODEL, SMALL, false, 0), "small");
    let whole = peak_of(&support::drawing(SHEETS, MODEL, WHOLE, false, 0), "whole");
    assert!(
        small * 2 <= whole,
        "a window onto 1/25 of the model peaks at {small} bytes against {whole} for all of it"
    );
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1713_a_budget_below_one_sheet_is_refused_by_code_and_writes_nothing() {
    let _g = measuring();
    let dir = dir_for("refuse");
    let dest = dir.join("plot.pdf");
    std::fs::write(&dest, b"existing").unwrap();
    let bytes = support::drawing(SHEETS, MODEL, SMALL, false, 0);
    let mut messages = Vec::new();
    for _ in 0..2 {
        let e = plot(&bytes, &dest, Some(limits(32 << 20)))
            .convert_with::<support::Synthetic>()
            .unwrap_err();
        assert_eq!(e.code(), "LIMIT_ALLOCATION_BYTES");
        messages.push(e.to_string());
    }
    assert!(
        messages[0].contains("S001"),
        "names the layout: {}",
        messages[0]
    );
    assert!(
        messages[0].contains("model"),
        "names the stage: {}",
        messages[0]
    );
    assert_eq!(messages[0], messages[1], "the same refusal every run");
    assert_eq!(std::fs::read(&dest).unwrap(), b"existing");
    let left: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name())
        .collect();
    assert_eq!(left.len(), 1, "nothing else is left behind: {left:?}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1713_a_budget_above_the_need_plots_the_same_bytes() {
    let _g = measuring();
    let dir = dir_for("same");
    let bytes = support::drawing(SHEETS, MODEL, SMALL, false, 0);
    let run = |name: &str, l: Option<SourceLimits>| {
        let dest = dir.join(name);
        plot(&bytes, &dest, l)
            .convert_with::<support::Synthetic>()
            .unwrap();
        std::fs::read(dest).unwrap()
    };
    let default = run("a.pdf", None);
    assert_eq!(default, run("b.pdf", Some(limits(1 << 30))));
    assert_eq!(default, run("c.pdf", Some(SourceLimits::raised())));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1713_leaving_out_what_is_off_the_window_does_not_change_the_page() {
    let _g = measuring();
    let dir = dir_for("cut");
    let run = |name: &str, only: bool| {
        let bytes = support::drawing(SHEETS, MODEL, SMALL, only, 0);
        let dest = dir.join(name);
        plot(&bytes, &dest, None)
            .convert_with::<support::Synthetic>()
            .unwrap();
        std::fs::read(dest).unwrap()
    };
    // Lines that cross the window's edge are in both, and are drawn cut.
    assert_eq!(run("all.pdf", false), run("window.pdf", true));
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1713_a_cancel_and_a_deadline_still_win_over_a_budget() {
    let _g = measuring();
    let dir = dir_for("stop");
    let bytes = support::drawing(SHEETS, MODEL, SMALL, false, 0);
    let token = CancelToken::new();
    token.cancel();
    let e = plot(&bytes, &dir.join("a.pdf"), Some(limits(8 << 20)))
        .cancel(token)
        .convert_with::<support::Synthetic>()
        .unwrap_err();
    assert_eq!(e.code(), "CANCELLED");
    let e = plot(&bytes, &dir.join("b.pdf"), Some(limits(8 << 20)))
        .deadline(Duration::ZERO)
        .convert_with::<support::Synthetic>()
        .unwrap_err();
    assert_eq!(e.code(), "DEADLINE");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1713_a_layout_set_records_the_refused_sheet_and_goes_on() {
    let _g = measuring();
    let dir = dir_for("set");
    // The last sheet shows the window through nine viewports, so it holds nine leaf sets.
    let bytes = support::drawing(SHEETS, 4_000, SMALL, false, 8);
    let out = PvecLayoutSet::new(CadSource::Bytes(&bytes), &dir)
        .stem("plan")
        .sheets(|c| c.unit(SourceUnit::Millimeters).limits(limits(11 << 19)))
        .convert_with::<support::Synthetic>()
        .unwrap();
    let failed: Vec<_> = out.failed().collect();
    assert_eq!(
        failed.len(),
        1,
        "only the nine viewport sheet is over budget"
    );
    assert_eq!(failed[0].layout, "S003");
    assert_eq!(out.written().count(), 2);
    assert_eq!(
        failed[0].outcome.as_ref().unwrap_err().code(),
        "LIMIT_ALLOCATION_BYTES"
    );
    let _ = std::fs::remove_dir_all(dir);
}
