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

pub mod checksum;
pub mod dedupe;
pub mod engine;
pub mod engine_builder;
pub mod extensions;
pub mod geo;
#[cfg(loom)]
mod loom_tests;
pub mod manifest;
pub mod observe;
pub mod pdf;
pub mod pixel;
pub mod planner;
pub mod raster;
pub mod resize;
pub mod resume;
pub mod retry;
pub mod sink;
#[cfg(feature = "s3")]
pub mod sink_object_store;
#[cfg(feature = "packfile")]
pub mod sink_packfile;
pub mod source;
pub mod streaming;
pub mod streaming_mapreduce;

// Curated crate-root surface: types and high-level entry points only.
// Leaf helpers, constants, and free functions stay behind their module path
// (e.g. `libviprs::resume::SCHEMA_VERSION`) so `use libviprs::*` does not
// flood callers with implementation detail.
pub use checksum::{ChecksumMode, VerifyError, VerifyReport};
pub use dedupe::{DedupeDecision, DedupeIndex, DedupeStrategy, LinkResult};
pub use engine::{
    BlankTileStrategy, EngineConfig, EngineError, EngineResult, StageDurations,
    generate_pyramid_resumable, is_blank_tile,
};
pub use engine_builder::{EngineBuilder, EngineKind, EngineSource, IntoEngineSource};
pub use geo::{GeoBounds, GeoCoord, GeoTransform, PixelCoord};
pub use manifest::{
    ChecksumAlgo, Checksums, GenerationSettings, LevelMetadata, Manifest, ManifestBuilder,
    ManifestError, ManifestV1, SourceMetadata, SparsePolicy,
};
pub use observe::{CollectingObserver, EngineEvent, EngineObserver, MemoryTracker};
#[cfg(feature = "pdfium")]
pub use pdf::{BudgetRenderResult, render_page_pdfium, render_page_pdfium_budgeted};
pub use pdf::{PdfError, PdfInfo, PdfPageInfo, extract_page_image, pdf_info};
pub use pixel::PixelFormat;
pub use planner::{
    Layout, LevelPlan, PlannerError, PyramidPlan, PyramidPlanner, TileCoord, TileRect,
};
pub use raster::{Raster, RasterError, RegionView};
pub use resume::{JobCheckpoint, JobMetadata, ResumeError, ResumeMode, ResumePolicy};
pub use retry::{FailurePolicy, RetryPolicy, RetryingSink};
pub use sink::{
    BLANK_TILE_MARKER, CollectedTile, FsSink, MemorySink, SinkError, Tile, TileFormat, TileSink,
};
#[cfg(feature = "s3")]
pub use sink_object_store::{ObjectStore, ObjectStoreConfig, ObjectStoreSink};
#[cfg(feature = "packfile")]
pub use sink_packfile::{PackfileFormat, PackfileSink, PackfileSinkBuilder};
pub use source::{SourceError, decode_bytes, decode_file, generate_test_raster};
#[cfg(feature = "pdfium")]
pub use streaming::PdfiumStripSource;
pub use streaming::{
    BudgetPolicy, RasterStripSource, StreamingConfig, StripSource, compute_strip_height,
    estimate_streaming_memory,
};
pub use streaming_mapreduce::MapReduceConfig;
