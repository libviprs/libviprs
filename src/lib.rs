#![cfg_attr(docsrs, feature(doc_cfg))]
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
//! 3. **Run** the pipeline with [`EngineBuilder::new`]`(source, plan, sink)`,
//!    chaining setters (e.g. `.with_engine(...)`, `.with_observer(...)`,
//!    `.with_concurrency(...)`) and finishing with `.run()`. The `source` is
//!    anything implementing [`IntoEngineSource`] — a `&Raster` or any
//!    [`StripSource`] — and the sink is typically an [`FsSink`] (filesystem) or
//!    [`MemorySink`] (in-memory).
//! 4. **Select an engine** with [`EngineKind`]: `Auto` (default; picks based on
//!    source kind and memory budget), `Monolithic` (in-memory),
//!    `Streaming` (sequential strip), `MapReduce` (parallel strip), or
//!    `MapReduceHotCache` (parallel strip, tiles cached in RAM and flushed to
//!    the sink in one canonical-order batch at the end).
//! 5. **Observe progress** by passing an [`EngineObserver`] to
//!    `.with_observer(...)`; lifecycle, level, tile, and batch updates arrive as
//!    [`EngineEvent`] variants (see the [`observe`] module).
//! 6. **Configure** blank tile handling with [`BlankTileStrategy`] to either emit
//!    full tiles or write 1-byte placeholders for uniform-color regions.
//!
//! ## Feature flags
//!
//! - **`pdfium`** — enables [`render_page_pdfium`], [`render_page_pdfium_budgeted`],
//!   and [`PdfiumStripSource`] for full vector PDF rendering via the pdfium library.
//! - **`pdfium-static`** — implies `pdfium` and statically links libpdfium.
//! - **`s3`** — gates the [`sink_object_store`] module ([`ObjectStoreSink`])
//!   against a user-injected [`ObjectStore`] backend.
//! - **`tracing`** — emits structured spans and events via the `tracing` crate.
//! - **`packfile`** — gates [`PackfileSink`] for writing tiles into tar or zip
//!   archives.
//!
//! ## Examples
//!
//! See the [libviprs-tests](https://github.com/libviprs/libviprs-tests) repository
//! for comprehensive integration tests, and
//! [libviprs-cli](https://github.com/libviprs/libviprs-cli) for a command-line
//! tool demonstrating every public API.
//!
//! **See also:** the [interactive CLI documentation](https://libviprs.org/cli/)
//! bundles every public knob into runnable examples.

pub mod arithmetic;
pub mod bands;
pub mod cancel;
pub mod checksum;
pub mod colour;
pub mod composite;
pub mod conversion;
pub mod dedupe;
pub mod draw;
pub mod engine;
pub mod engine_builder;
pub mod extensions;
pub mod extract;
pub mod geo;
pub mod histogram;
pub mod imageio;
pub(crate) mod level_walk;
#[cfg(loom)]
mod loom_tests;
pub mod manifest;
pub(crate) mod mapreduce_hot_cache;
pub mod observe;
pub mod pdf;
pub mod pixel;
pub mod planner;
pub(crate) mod poison;
pub mod raster;
pub(crate) mod raster_ops;
pub mod resize;
pub mod resume;
pub mod retry;
pub mod sink;
#[cfg(feature = "s3")]
#[cfg_attr(docsrs, doc(cfg(feature = "s3")))]
pub mod sink_object_store;
#[cfg(feature = "packfile")]
#[cfg_attr(docsrs, doc(cfg(feature = "packfile")))]
pub mod sink_packfile;
pub mod source;
pub mod stream_verify;
pub mod streaming;
pub mod streaming_mapreduce;
pub(crate) mod sync_queue;
pub mod verify;

// Curated crate-root surface: types and high-level entry points only.
// Leaf helpers, constants, and free functions stay behind their module path
// (e.g. `libviprs::resume::SCHEMA_VERSION`) so `use libviprs::*` does not
// flood callers with implementation detail.
pub use arithmetic::{ArithmeticError, Comparand};
pub use bands::BandError;
pub use cancel::CancelToken;
pub use checksum::{ChecksumMode, VerifyError, VerifyReport};
pub use colour::{ColourError, Intent, Pcs};
pub use composite::{CompositeError, CompositeMode};
pub use conversion::{Angle, Angle45, ConversionError, Interpretation, RasterCopyBuilder};
pub use dedupe::{DedupeDecision, DedupeIndex, DedupeStrategy, LinkResult};
pub use draw::{Circle, DrawError, DrawOp, Flood, Line, Mask, Paste, Rectangle, Smudge};
pub use engine::{
    BlankTileStrategy, EngineConfig, EngineError, EngineResult, StageDurations, is_blank_tile,
};
pub use engine_builder::{EngineBuilder, EngineKind, EngineSource, IntoEngineSource};
pub use extract::{CompassDirection, Extend, ExtractError, SmartcropInteresting};
pub use geo::{GeoBounds, GeoCoord, GeoTransform, PixelCoord};
pub use histogram::HistogramError;
// The imageio free functions are re-exported at the root (not just behind
// the module path) because the ported tests import them from the crate
// root (`use libviprs::{tokenize, get_max_coord, ...}`).
pub use imageio::{
    MetadataError, MetadataValue, SaveError, ThumbnailGeometry, get_max_coord, init_from_env,
    parse_thumbnail_geometry, set_max_coord, tokenize,
};
pub use manifest::{
    ChecksumAlgo, Checksums, GenerationSettings, LevelMetadata, Manifest, ManifestBuilder,
    ManifestError, ManifestV1, SourceMetadata, SparsePolicy,
};
pub use observe::{
    CollectingObserver, EngineEvent, EngineObserver, FanOutObserver, MemoryTracker, WorkerId,
};
#[cfg(feature = "pdfium")]
#[cfg_attr(docsrs, doc(cfg(feature = "pdfium")))]
pub use pdf::{BudgetRenderResult, render_page_pdfium, render_page_pdfium_budgeted};
pub use pdf::{PageRotation, PdfError, PdfInfo, PdfPageInfo, extract_page_image, pdf_info};
pub use pixel::PixelFormat;
pub use planner::{
    Layout, LevelPlan, PlannerError, PyramidPlan, PyramidPlanner, TileCoord, TileRect,
};
pub use raster::{Raster, RasterError, RegionView};
pub use resume::{
    CompletedTileSet, JobCheckpoint, JobMetadata, ResumeError, ResumeMode, ResumePolicy,
};
pub use retry::{FailurePolicy, RetryPolicy, RetryingSink};
pub use sink::{
    BLANK_TILE_MARKER, CollectedTile, FsSink, MemorySink, SinkError, Tile, TileFormat, TileSink,
};
#[cfg(feature = "s3")]
#[cfg_attr(docsrs, doc(cfg(feature = "s3")))]
pub use sink_object_store::{ObjectStore, ObjectStoreConfig, ObjectStoreSink};
#[cfg(feature = "packfile")]
#[cfg_attr(docsrs, doc(cfg(feature = "packfile")))]
pub use sink_packfile::{PackfileFormat, PackfileSink, PackfileSinkBuilder};
pub use source::{SourceError, decode_bytes, decode_file, generate_test_raster};
pub use streaming::{
    BudgetPolicy, RasterStripSource, StreamingConfig, StripSource, compute_strip_height,
    estimate_streaming_memory,
};
#[cfg(feature = "pdfium")]
#[cfg_attr(docsrs, doc(cfg(feature = "pdfium")))]
pub use streaming::{PdfiumRenderMode, PdfiumStripSource};
pub use streaming_mapreduce::{
    LocalWorkExecutor, MapReduceConfig, StripWorkUnit, WorkContext, WorkExecutor,
};
