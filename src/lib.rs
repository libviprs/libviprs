//! # libviprs — High-performance tile pyramid generation
//!
//! libviprs converts large raster images and PDF documents into multi-resolution
//! tile pyramids suitable for Deep Zoom viewers, slippy-map UIs, and GIS applications.
//!
//! ## Core workflow
//!
//! 1. **Load** an image with [`decode_file`] / [`decode_bytes`], or extract from
//!    PDF with [`extract_page_image`] / [`render_page_pdfium`].
//! 2. **Plan** the pyramid with [`PyramidPlanner`] — choose tile size, overlap,
//!    and layout ([`Layout::DeepZoom`] or [`Layout::Xyz`]).
//! 3. **Generate** tiles with [`generate_pyramid`] or [`generate_pyramid_observed`],
//!    sending output to an [`FsSink`] (filesystem) or [`MemorySink`] (in-memory).
//! 4. **Configure** blank tile handling with [`BlankTileStrategy`] to either emit
//!    full tiles or write 1-byte placeholders for uniform-color regions.
//!
//! ## Feature flags
//!
//! - **`pdfium`** — enables [`render_page_pdfium`] and [`render_page_pdfium_budgeted`]
//!   for full vector PDF rendering via the pdfium library.
//!
//! ## Examples
//!
//! See the [libviprs-tests](https://github.com/libviprs/libviprs-tests) repository
//! for comprehensive integration tests, and
//! [libviprs-cli](https://github.com/libviprs/libviprs-cli) for a command-line
//! tool demonstrating every public API.

pub mod engine;
pub mod geo;
#[cfg(loom)]
mod loom_tests;
pub mod observe;
pub mod pdf;
pub mod pixel;
pub mod planner;
pub mod raster;
pub mod resize;
pub mod sink;
pub mod source;
pub mod streaming;

pub use engine::{
    BlankTileStrategy, EngineConfig, EngineError, EngineResult, generate_pyramid,
    generate_pyramid_observed, is_blank_tile,
};
pub use geo::{GeoBounds, GeoCoord, GeoTransform, PixelCoord};
pub use observe::{CollectingObserver, EngineEvent, EngineObserver, MemoryTracker};
#[cfg(feature = "pdfium")]
pub use pdf::{BudgetRenderResult, render_page_pdfium, render_page_pdfium_budgeted};
pub use pdf::{PdfError, PdfInfo, PdfPageInfo, extract_page_image, pdf_info};
pub use pixel::PixelFormat;
pub use planner::{
    Layout, LevelPlan, PlannerError, PyramidPlan, PyramidPlanner, TileCoord, TileRect,
};
pub use raster::{Raster, RasterError, RegionView};
pub use sink::{BLANK_TILE_MARKER, FsSink, MemorySink, SinkError, Tile, TileFormat, TileSink};
pub use source::{SourceError, decode_bytes, decode_file, generate_test_raster};
#[cfg(feature = "pdfium")]
pub use streaming::PdfiumStripSource;
pub use streaming::{
    RasterStripSource, StreamingConfig, StripSource, compute_strip_height,
    estimate_streaming_memory, generate_pyramid_auto, generate_pyramid_streaming,
};
