//! Issue #1726: a plot's memory budget counts what the process holds.
//!
//! A plot charged only the model pass, at a flat 1 KiB a leaf, and nothing for the drawing it had
//! opened, so a 1.5 GiB budget let a real peak of 2 GiB through and a 512 MiB budget refused only
//! after the process already held 1.8 GiB. These tests measure the peak as live heap bytes with a
//! counting global allocator, as `pvec_1713_plot_cull.rs` does, on a synthetic drawing whose opened
//! form really holds memory (24 times its source, the ratio of a real DWG), and check two things:
//! the estimate the budget is checked against tracks the real peak, and a budget that is too small
//! refuses before the process holds more than the budget. The counter is process wide, so every test
//! takes [`MEASURING`]. Needs `--features pvec-convert`.

#![cfg(feature = "pvec-convert")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};

use libviprs::cad::CadSource;
use libviprs::pvec::{
    DwgProvider, EntityCursor, EntityScope, LayerState, LayoutPlot, LayoutState, ProviderCaps,
    ProviderError, SourceFact, SourceLimits, SourceUnit, ViewProfile, ViewportEntity,
};
use libviprs::pvec_convert::{PdfPlot, PvecConvertError};

#[path = "pvec_1713_support/mod.rs"]
mod synthetic;

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

/// What an opened drawing holds for each byte of its file. A real DWG measured 24 times its size
/// once opened (28.5 MiB held 688 MiB).
const OPENED_PER_BYTE: usize = 24;

/// The synthetic drawing of #1713, opened so that it holds [`OPENED_PER_BYTE`] times its source, the
/// way a decoded DWG does. The source is the drawing's description followed by padding.
struct Opened {
    inner: synthetic::Synthetic,
    _held: Vec<u8>,
}

impl DwgProvider for Opened {
    fn probe(source: CadSource<'_>) -> Result<bool, ProviderError> {
        synthetic::Synthetic::probe(source)
    }
    fn open(source: CadSource<'_>) -> Result<Self, ProviderError> {
        let len = match source {
            CadSource::Bytes(b) => b.len(),
            _ => 0,
        };
        Ok(Self {
            inner: synthetic::Synthetic::open(source)?,
            _held: vec![1u8; len * OPENED_PER_BYTE],
        })
    }
    fn name(&self) -> &'static str {
        self.inner.name()
    }
    fn caps(&self) -> &ProviderCaps {
        self.inner.caps()
    }
    fn layouts(&self) -> Result<Vec<LayoutState>, ProviderError> {
        self.inner.layouts()
    }
    fn layout_plot(
        &self,
        layout: libviprs::pvec::LayoutId,
    ) -> Result<SourceFact<LayoutPlot>, ProviderError> {
        self.inner.layout_plot(layout)
    }
    fn layers(&self) -> Result<Vec<LayerState>, ProviderError> {
        self.inner.layers()
    }
    fn viewports(&self) -> Result<Vec<ViewportEntity>, ProviderError> {
        self.inner.viewports()
    }
    fn entities<'a>(
        &'a self,
        scope: &EntityScope,
    ) -> Result<Box<dyn EntityCursor + 'a>, ProviderError> {
        self.inner.entities(scope)
    }
}

const SHEETS: u32 = 3;
const MODEL: u32 = 60_000;
const SMALL: u32 = 28;
/// The source is padded to this size, so the opened drawing holds 60 MiB.
const PADDED: usize = 5 << 19;

/// The drawing's bytes, padded; the padding comes after the last field the mock reads.
fn drawing() -> Vec<u8> {
    let mut bytes = synthetic::drawing(SHEETS, MODEL, SMALL, false, 0);
    bytes.push(b'-');
    bytes.resize(PADDED, b'x');
    bytes
}

fn limits(bytes: u64) -> SourceLimits {
    let mut l = SourceLimits::default();
    l.max_allocation_bytes = bytes;
    l
}

fn dir_for(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("libviprs-1726-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A plot with `budget` (the default when `None`): the live heap it held at most above what was
/// live before it started, and how it ended.
fn plot(bytes: &[u8], dest: &Path, budget: Option<u64>) -> (u64, Result<(), PvecConvertError>) {
    let before = LIVE.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let mut p = PdfPlot::new(CadSource::Bytes(bytes), dest)
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("S001"))
        .layouts((1..=SHEETS).map(|n| format!("S{n:03}")));
    if let Some(b) = budget {
        p = p.limits(limits(b));
    }
    let done = p.convert_with::<Opened>().map(|_| ());
    (PEAK.load(Ordering::Relaxed).saturating_sub(before), done)
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1726_a_budget_below_the_real_peak_refuses_instead_of_plotting() {
    let _g = measuring();
    let dir = dir_for("estimate");
    let bytes = drawing();
    let (peak, done) = plot(&bytes, &dir.join("a.pdf"), None);
    done.unwrap();
    // The estimate may sit a quarter under the real peak and no further: a budget at four fifths of
    // what the plot really holds must refuse. Counting the model pass alone, with the drawing it
    // was read from left out, lets this one plot.
    let budget = peak / 5 * 4;
    let (_, done) = plot(&bytes, &dir.join("b.pdf"), Some(budget));
    let e = done.expect_err("a budget under the real peak plots anyway");
    assert_eq!(e.code(), "LIMIT_ALLOCATION_BYTES", "{e}");
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1726_a_budget_above_the_real_peak_plots_the_same_bytes() {
    let _g = measuring();
    let dir = dir_for("above");
    let bytes = drawing();
    let (peak, done) = plot(&bytes, &dir.join("a.pdf"), None);
    done.unwrap();
    // The estimate is not allowed to run far over either: half as much again as the real peak is
    // enough, and the page is the one the default budget gives.
    let (_, done) = plot(&bytes, &dir.join("b.pdf"), Some(peak / 2 * 3));
    done.unwrap();
    assert_eq!(
        std::fs::read(dir.join("a.pdf")).unwrap(),
        std::fs::read(dir.join("b.pdf")).unwrap()
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1726_a_budget_below_the_opened_drawing_refuses_before_opening_it() {
    let _g = measuring();
    let dir = dir_for("open");
    let bytes = drawing();
    // Half of what the opened drawing holds: the file is on disk or in hand, and its size alone says
    // that opening it would break the budget.
    let budget = (PADDED * OPENED_PER_BYTE / 2) as u64;
    let (peak, done) = plot(&bytes, &dir.join("a.pdf"), Some(budget));
    let e = done.expect_err("the plot should be refused");
    assert_eq!(e.code(), "LIMIT_ALLOCATION_BYTES", "{e}");
    assert!(e.to_string().contains("drawing"), "names the stage: {e}");
    assert!(
        peak <= budget,
        "the process held {peak} bytes before it refused a budget of {budget}"
    );
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1726_a_budget_that_fits_the_drawing_but_not_the_model_refuses_while_reading_it() {
    let _g = measuring();
    let dir = dir_for("model");
    let bytes = drawing();
    // Room for the opened drawing and a few leaves of the model: the walk must stop when the leaves
    // it holds reach the budget, not after the whole model is held.
    let budget = (PADDED * 30 + (4 << 20)) as u64;
    let (peak, done) = plot(&bytes, &dir.join("a.pdf"), Some(budget));
    let e = done.expect_err("the plot should be refused");
    assert_eq!(e.code(), "LIMIT_ALLOCATION_BYTES", "{e}");
    assert!(
        peak <= budget,
        "the process held {peak} bytes before it refused a budget of {budget}"
    );
    let _ = std::fs::remove_dir_all(dir);
}
