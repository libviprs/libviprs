//! Issue #1466: a memory budget is silently not honoured by the cached
//! `PdfiumStripSource`, which is what `new()` and the builder's default give.
//!
//! The cached mode rasterises the whole page when it is built and keeps it. The
//! strip budget (`new_with_budget`, `.budget(..)`) only looks at the worst-case
//! strip and `with_memory_budget` only governs the engine's strips, so a page
//! whose full raster is many times the budget passes both. These tests are red
//! until the cached raster counts against the budget.
//!
//! The page is a 1000 pt blank square: 4167 x 4167 px at 300 DPI, a 66 MiB
//! raster, against a 32 MiB budget. A strip of it is 4 MiB, so only the cached
//! raster is over. They need a libpdfium at runtime (`PDFIUM_PATH` or the system
//! library), the same as the other pdfium suites.
#![cfg(feature = "pdfium")]

use std::path::{Path, PathBuf};

use libviprs::{
    BudgetPolicy, EngineBuilder, EngineKind, Layout, MemorySink, PdfError, PdfiumRenderMode,
    PdfiumStripSource, PyramidPlanner, StripSource,
};

const BUDGET: u64 = 32 * 1024 * 1024;

fn blank_page(tag: &str) -> PathBuf {
    use lopdf::{Document, Object, Stream, dictionary};

    let mut doc = Document::with_version("1.5");
    let pages_id = doc.new_object_id();
    let content_id = doc.add_object(Stream::new(dictionary! {}, Vec::new()));
    let page_id = doc.add_object(dictionary! {
        "Type" => "Page",
        "Parent" => pages_id,
        "Contents" => content_id,
        "MediaBox" => vec![0.into(), 0.into(), 1000.into(), 1000.into()],
    });
    doc.objects.insert(
        pages_id,
        Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Kids" => vec![page_id.into()],
            "Count" => 1,
        }),
    );
    let catalog_id = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
    doc.trailer.set("Root", catalog_id);
    let path = std::env::temp_dir().join(format!("viprs-1466-{}-{tag}.pdf", std::process::id()));
    doc.save(&path).expect("write the blank page");
    path
}

fn raster_bytes(source: &PdfiumStripSource) -> u64 {
    u64::from(source.width()) * u64::from(source.height()) * 4
}

/// A cached source asked to stay inside a budget its raster is twice over must
/// say so, the way a strip over the budget already does.
#[test]
fn a_cached_source_over_the_budget_is_refused_under_the_error_policy() {
    let pdf = blank_page("refused");
    let built = PdfiumStripSource::builder(&pdf, 1, 300)
        .mode(PdfiumRenderMode::CachedFullPage)
        .budget(256, BUDGET, BudgetPolicy::Error)
        .build();
    match built {
        Err(PdfError::BudgetExceeded { .. }) => {}
        Err(other) => panic!("expected BudgetExceeded, got {other}"),
        Ok(source) => panic!(
            "built a cached source holding {} MiB under a {} MiB budget",
            raster_bytes(&source) >> 20,
            BUDGET >> 20
        ),
    }
}

/// Under `AutoAdjustDpi` the DPI comes down until the whole cached raster fits.
#[test]
fn a_cached_source_under_auto_adjust_dpi_is_brought_inside_the_budget() {
    let pdf = blank_page("adjusted");
    let source = PdfiumStripSource::builder(&pdf, 1, 300)
        .mode(PdfiumRenderMode::CachedFullPage)
        .budget(256, BUDGET, BudgetPolicy::AutoAdjustDpi { min_dpi: 72 })
        .build()
        .expect("a page that fits at a lower DPI should build");
    assert!(
        raster_bytes(&source) <= BUDGET,
        "the cached raster is {} MiB, over the {} MiB budget, at {} dpi",
        raster_bytes(&source) >> 20,
        BUDGET >> 20,
        source.dpi()
    );
}

/// The engine is handed a `with_memory_budget` and a source that already holds
/// more than that. Running anyway is the silent case from the report.
#[test]
fn the_engine_does_not_run_a_cached_source_that_alone_is_over_its_memory_budget() {
    let pdf: &Path = &blank_page("engine");
    let source = PdfiumStripSource::new(pdf, 1, 300).expect("open the page");
    assert!(
        raster_bytes(&source) > 2 * BUDGET,
        "the page is meant to be well over"
    );
    let plan = PyramidPlanner::new(source.width(), source.height(), 256, 0, Layout::Google)
        .expect("plan")
        .plan();

    let result = EngineBuilder::new(source, plan, MemorySink::new())
        .with_engine(EngineKind::Streaming)
        .with_memory_budget(BUDGET)
        .with_budget_policy(BudgetPolicy::Error)
        .run();

    assert!(
        result.is_err(),
        "the run finished inside a {} MiB budget with a source holding 66 MiB",
        BUDGET >> 20
    );
}
