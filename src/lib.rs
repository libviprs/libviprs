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

pub use engine::{
    EngineConfig, EngineError, EngineResult, generate_pyramid, generate_pyramid_observed,
    is_blank_tile,
};
pub use geo::{GeoBounds, GeoCoord, GeoTransform, PixelCoord};
pub use observe::{CollectingObserver, EngineEvent, EngineObserver, MemoryTracker};
#[cfg(feature = "pdfium")]
pub use pdf::render_page_pdfium;
pub use pdf::{PdfError, PdfInfo, PdfPageInfo, extract_page_image, pdf_info};
pub use pixel::PixelFormat;
pub use planner::{
    Layout, LevelPlan, PlannerError, PyramidPlan, PyramidPlanner, TileCoord, TileRect,
};
pub use raster::{Raster, RasterError, RegionView};
pub use sink::{FsSink, MemorySink, SinkError, Tile, TileFormat, TileSink};
pub use source::{SourceError, decode_bytes, decode_file, generate_test_raster};
