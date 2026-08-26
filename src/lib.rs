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
//! - **`object-store-sink`** — gates the [`sink_object_store`] module
//!   ([`ObjectStoreSink`]) against a user-injected [`ObjectStore`] backend. The
//!   former name **`s3`** is retained as a deprecated alias
//!   (`s3 = ["object-store-sink"]`) that enables the same module; prefer
//!   `object-store-sink`, as the `s3` alias will be removed in a future release.
//! - **`tracing`** — emits structured spans and events via the `tracing` crate.
//! - **`packfile`** — gates [`PackfileSink`] for writing tiles into tar or zip
//!   archives.
//!
//! ## Error handling and the dual API
//!
//! Image operations come in two forms, and this pairing is a deliberate design
//! choice rather than incidental duplication:
//!
//! - a **fallible** `try_*` method returning `Result<_, `*module*`Error>` —
//!   the primary form for production code, surfacing data-dependent failures
//!   (dimension / band-count mismatches, float input to an integer op,
//!   out-of-range indices) as typed errors; and
//! - a **panicking** short name (`add`, `sub`, `rem_const`, ...) that delegates
//!   to the `try_*` form and `expect`s the result — an ergonomic convenience
//!   for tests, examples, and call sites where the inputs are statically known
//!   to be valid. Each carries a `# Panics` section naming exactly what its
//!   fallible twin rejects.
//!
//! Operations that cannot fail on the value domain — the whole-image
//! reductions and the float-output `linear` / `div` family
//! ([`Raster::div_const`], [`Raster::linear`], [`Raster::linear_uchar`]),
//! which accept every input depth and always produce a representable result —
//! expose only the single infallible form.
//!
//! "Cannot fail on the value domain" is the exact claim, and it is narrower
//! than "cannot fail". These infallible forms still allocate a
//! dimension-sized output, and having no error channel they *panic* when the
//! allocator cannot satisfy that request — an oversized-raster failure driven
//! by the image dimensions, never by the sample values. Their `# Panics`
//! contract is therefore the same allocation panic the panicking twins carry;
//! only the value-domain path is guaranteed infallible.
//!
//! Each op module owns a typed error enum ([`ArithmeticError`], [`BandError`],
//! ...), which keeps a single-family caller's error surface tight. A caller
//! *composing* several families can funnel them all through the
//! `#[non_exhaustive]` crate-level [`OpError`] umbrella and lean on `?` without
//! bespoke mapping glue (see the [`error`] module).
//!
//! The umbrella's boundary is deliberate, not exhaustive: it carries a
//! `#[from]` conversion for each **in-memory pixel-transform** op-family error
//! (arithmetic, bands, colour, composite, convolution, conversion, create,
//! draw, extract, freqfilt, histogram, matrix, morphology, mosaicing, resample,
//! plus the core [`Raster`] error). The I/O, codec, and pipeline error families
//! (source / sink / save / metadata, encode, engine / planner / resume /
//! manifest) are intentionally **excluded** — they belong to different call
//! surfaces than the pixel-transform ops and are not funnelled here.
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
pub mod codec;
pub mod colour;
pub mod composite;
pub mod connection;
pub mod conversion;
pub mod convolution;
pub mod create;
pub mod dedupe;
pub mod draw;
pub mod encode;
pub mod encode_tiff;
pub mod engine;
pub mod engine_builder;
pub mod error;
pub mod extensions;
pub mod extract;
pub mod foreign_stubs;
pub mod freqfilt;
pub mod geo;
pub mod gif;
pub(crate) mod hex;
pub mod histogram;
pub mod imageio;
pub(crate) mod level_walk;
#[cfg(loom)]
mod loom_checkpoint_dedupe;
#[cfg(loom)]
mod loom_tests;
pub mod manifest;
pub(crate) mod mapreduce_hot_cache;
pub mod matrix;
pub mod morphology;
pub mod mosaicing;
pub mod observe;
pub mod pdf;
pub mod pixel;
pub mod planner;
pub(crate) mod poison;
pub mod radiance;
pub mod raster;
pub(crate) mod raster_ops;
pub mod resample;
pub mod resize;
pub mod resume;
pub mod retry;
pub mod sink;
#[cfg(feature = "object-store-sink")]
#[cfg_attr(docsrs, doc(cfg(feature = "object-store-sink")))]
pub mod sink_object_store;
#[cfg(feature = "packfile")]
#[cfg_attr(docsrs, doc(cfg(feature = "packfile")))]
pub mod sink_packfile;
pub mod source;
pub mod stream_verify;
pub mod streaming;
pub mod streaming_mapreduce;
pub mod svg;
pub(crate) mod sync_queue;
pub mod textio;
pub mod verify;
pub mod webp;

// Curated crate-root surface: types and high-level entry points only.
// Leaf helpers, constants, and free functions stay behind their module path
// (e.g. `libviprs::resume::SCHEMA_VERSION`) so `use libviprs::*` does not
// flood callers with implementation detail.
pub use arithmetic::{ArithmeticError, Comparand};
pub use bands::BandError;
pub use cancel::CancelToken;
pub use checksum::{ChecksumMode, VerifyError, VerifyReport};
pub use codec::{DecodeError, EncodeError, JpegSubsample, TiffCompression};
pub use colour::{ColourError, Intent, Pcs};
pub use composite::{CompositeError, CompositeMode};
pub use connection::{Source, Target, decode_source, encode_to_target};
pub use conversion::{
    Align, Angle, Angle45, ConversionError, Interpretation, JoinDirection, RasterCopyBuilder,
};
pub use convolution::{Combine, ConvolutionError, Kernel, Precision};
pub use create::{CreateError, SdfParams};
pub use dedupe::{DedupeDecision, DedupeIndex, DedupeStrategy, LinkResult};
pub use draw::{Circle, DrawError, DrawOp, Flood, Line, Mask, Paste, Rectangle, Smudge};
// The TIFF free functions are re-exported at the root (not just behind the
// module path) because the ported foreign cells call them unqualified
// (`tiff_page_count(...)`, `decode_tiff_page(...)`). The `_with_limits` twins
// come with them: they are the same two entry points with the resource
// ceilings passed in rather than defaulted, and splitting a pair across two
// import paths would only make the bounded form the harder one to reach. The
// `save_tiff` family and the `tiff_save` / `tiff_load` round-trip are inherent
// methods on `Raster` and travel with the already-exported `Raster` type.
pub use encode_tiff::{
    decode_tiff_page, decode_tiff_page_with_limits, tiff_page_count, tiff_page_count_with_limits,
};
pub use engine::{
    BlankTileStrategy, EngineConfig, EngineError, EngineResult, StageDurations,
    generate_pyramid_region, is_blank_tile,
};
pub use engine_builder::{EngineBuilder, EngineKind, EngineSource, IntoEngineSource};
pub use error::OpError;
pub use extract::{CompassDirection, Extend, ExtractError, SmartcropInteresting};
// The deferred foreign-format free functions and options are re-exported at
// the root because the ported foreign cell reaches them there
// (`use libviprs::{magickload, decode_openslide, ...}`), matching the imageio
// convention above. The deferred encoders and `dzsave_buffer` are inherent
// methods on `Raster`, so they need no re-export. `decode_svg` used to live
// here too; it moved to `crate::svg` when the SVG lane made it real
// (issue #502) and is re-exported from there, so the crate-root spelling is
// unchanged.
pub use foreign_stubs::{
    MagickLoadOptions, decode_bytes_fail_on, decode_file_fail_on, decode_openslide, magickload,
    magickload_with,
};
pub use freqfilt::FreqfiltError;
pub use geo::{GeoBounds, GeoCoord, GeoTransform, PixelCoord};
pub use histogram::HistogramError;
// The imageio free functions are re-exported at the root (not just behind
// the module path) because the ported tests import them from the crate
// root (`use libviprs::{tokenize, parse_thumbnail_geometry, ...}`).
// `decode_svg` keeps its crate-root spelling because the ported foreign and
// connection cells import it from there (`use libviprs::{decode_svg, ...}`),
// the same reason the deferred foreign free functions are re-exported above.
// `SvgOptions` travels with it so a caller never has to name the module path
// just to build the argument.
pub use imageio::{
    MetadataError, MetadataValue, SaveError, ThumbnailGeometry, parse_thumbnail_geometry, tokenize,
};
pub use manifest::{
    ChecksumAlgo, Checksums, GenerationSettings, LevelMetadata, Manifest, ManifestBuilder,
    ManifestError, ManifestV1, SourceMetadata, SparsePolicy,
};
pub use matrix::MatrixError;
pub use morphology::{Direction, MorphologyError};
pub use mosaicing::{MergeDirection, MosaicError};
pub use observe::{
    CollectingObserver, EngineEvent, EngineObserver, FanOutObserver, MemoryTracker, WorkerId,
};
pub use pdf::{
    BackgroundColor, PageRotation, PdfError, PdfInfo, PdfPageInfo, extract_page_image,
    extract_page_image_dpi, extract_page_image_with_background,
    extract_page_image_with_background_typed, extract_page_image_with_password, pdf_info,
    pdf_info_with_password,
};
#[cfg(feature = "pdfium")]
#[cfg_attr(docsrs, doc(cfg(feature = "pdfium")))]
pub use pdf::{BudgetRenderResult, render_page_pdfium, render_page_pdfium_budgeted};
pub use pixel::PixelFormat;
pub use planner::{
    Layout, LevelPlan, PlannerError, PyramidPlan, PyramidPlanner, TileCoord, TileRect,
};
// `decode_radiance` is re-exported beside the error type because it is the
// format-specific decode entry point a caller reaches for when they already
// know the bytes are Radiance, exactly as `decode_tiff_page` is; the
// content-sniffing `decode_bytes` / `decode_file` reach it too.
pub use radiance::{RadianceError, decode_radiance};
// `decode_gif` is re-exported beside its error type for the same reason
// `decode_radiance` is: it is the direct entry point for a caller who
// already knows the bytes are a GIF and does not want to go through the
// sniff route.
pub use gif::{GifError, decode_gif};
pub use raster::{Raster, RasterError, RegionView};
pub use resample::{
    AffineOptions, Interpolator, ReduceKernel, ResampleError, ResizeOptions, ThumbnailError,
    thumbnail, thumbnail_crop,
};
pub use resume::{
    CompletedTileSet, JobCheckpoint, JobMetadata, ResumeError, ResumeMode, ResumePolicy,
};
pub use retry::{FailurePolicy, RetryPolicy, RetryingSink};
pub use sink::{
    BLANK_TILE_MARKER, CollectedTile, FsSink, MemorySink, SinkError, Tile, TileFormat, TileSink,
};
#[cfg(feature = "object-store-sink")]
#[cfg_attr(docsrs, doc(cfg(feature = "object-store-sink")))]
pub use sink_object_store::{ObjectStore, ObjectStoreConfig, ObjectStoreSink};
#[cfg(feature = "packfile")]
#[cfg_attr(docsrs, doc(cfg(feature = "packfile")))]
pub use sink_packfile::{PackfileFormat, PackfileSink, PackfileSinkBuilder, ZipSink};
pub use source::{
    SourceError, clear_load_cache, decode_bytes, decode_file, decode_file_sequential,
    decode_file_with_options, decode_file_with_shrink, generate_test_raster,
    set_load_cache_max_bytes, set_load_cache_max_entries,
};
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
pub use svg::{SvgOptions, decode_svg, decode_svg_with_limits};
// `decode_webp` is re-exported for the reason `decode_radiance` is: it is
// the format-specific decode entry point a caller reaches for when they
// already know the bytes are WebP. The option types stay behind
// `libviprs::webp::` so the crate root does not gain a second `SaveOptions`.
pub use webp::decode_webp;
// The text/tabular decoders are inherent associated functions on `Raster`
// (`Raster::matrix_load`, `Raster::csv_load`, `Raster::ppm_load`), so the
// ported connection and foreign cells reach them through the crate-root
// `Raster` re-export above. Associated functions need no free-function
// re-export, so the crate root exposes only their encoder counterparts on
// `Raster` and the `textio` module path itself.
