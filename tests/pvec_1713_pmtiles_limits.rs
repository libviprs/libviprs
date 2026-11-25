//! Issue #1713: a PMTiles plot charges the raster of each page against `max_allocation_bytes` before
//! it is rendered, and refuses by code with nothing written. Needs
//! `--features pvec-convert,pdfium` (libpdfium at runtime).

#![cfg(all(feature = "pvec-convert", feature = "pdfium"))]

use libviprs::cad::CadSource;
use libviprs::pvec::{SourceLimits, SourceUnit, ViewProfile};
use libviprs::pvec_convert::{PmTilesPlot, PmTilesPlotError};

#[path = "pvec_1713_support/mod.rs"]
mod support;

#[test]
#[cfg_attr(miri, ignore)]
fn issue_1713_a_raster_over_the_budget_is_refused_before_it_is_rendered() {
    let dir = std::env::temp_dir().join(format!("libviprs-1713-tiles-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let bytes = support::drawing(1, 10, 28, false, 0);
    // A 420 by 297 mm sheet at 300 dpi is about 70 MB of RGBA, and ten lines are a few kilobytes.
    let mut limits = SourceLimits::default();
    limits.max_allocation_bytes = 8 << 20;
    let err = PmTilesPlot::new(CadSource::Bytes(&bytes), &dir)
        .unit(SourceUnit::Millimeters)
        .view_profile(ViewProfile::print_profile("S001"))
        .source_limits(limits)
        .convert_with::<support::Synthetic>()
        .unwrap_err();
    let PmTilesPlotError::Plot(e) = err else {
        panic!("expected a refused plot, got {err:?}");
    };
    assert_eq!(e.code(), "LIMIT_ALLOCATION_BYTES");
    assert!(
        e.to_string().contains("raster") && e.to_string().contains("S001"),
        "{e}"
    );
    let left: Vec<_> = std::fs::read_dir(&dir).unwrap().collect();
    assert!(left.is_empty(), "nothing is written: {left:?}");
    let _ = std::fs::remove_dir_all(dir);
}
