//! Plot to PMTiles (#1676): one raster pyramid archive per plotted sheet.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use acadrust::plot_style::PlotStyleTable;

use super::{PlotSettings, PvecConvertError};
use crate::CancelToken;
use crate::cad::CadSource;
use crate::observe::EngineObserver;
use crate::pvec::{DwgProvider, SourceUnit, ViewProfile};

/// Bounds on what one sheet may ask of the raster path.
#[derive(Debug, Clone, PartialEq)]
pub struct RasterLimits {
    /// Longest page side, in points.
    pub max_page_pts: f64,
    /// Most pixels in the rendered page.
    pub max_pixels: u64,
    /// Most zoom levels in one archive.
    pub max_zoom_levels: u32,
    /// Most tiles in one archive.
    pub max_tiles: u64,
}

impl Default for RasterLimits {
    fn default() -> Self {
        Self {
            max_page_pts: 14_400.0,
            max_pixels: 1 << 28,
            max_zoom_levels: 16,
            max_tiles: 100_000,
        }
    }
}

/// One archive.
#[derive(Debug, Clone)]
pub struct PmTilesSheet {
    /// The layout it plots.
    pub layout: String,
    /// The archive.
    pub file: PathBuf,
    /// Raster width in pixels.
    pub width_px: u32,
    /// Raster height in pixels.
    pub height_px: u32,
    /// Lowest zoom.
    pub min_zoom: u8,
    /// Highest zoom.
    pub max_zoom: u8,
    /// Tiles written.
    pub tiles: u64,
}

/// What a PMTiles plot wrote.
#[derive(Debug)]
pub struct PmTilesPlotOutput {
    /// One per sheet, in the order asked for.
    pub sheets: Vec<PmTilesSheet>,
    /// The plot's notes.
    pub notes: Vec<String>,
}

/// Why a PMTiles plot failed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PmTilesPlotError {
    /// Not written yet.
    #[error("not written yet")]
    NotYet,
    /// The plot itself failed.
    #[error("the plot failed")]
    Plot(#[source] PvecConvertError),
    /// A page is longer than the limit.
    #[error("page {layout} is {long_pts} pt long, the limit is {max_pts}")]
    PageTooLarge {
        /// The layout.
        layout: String,
        /// Its long side.
        long_pts: f64,
        /// The limit.
        max_pts: f64,
    },
    /// The dpi is outside what is allowed.
    #[error("dpi {0} is outside 36 to 600")]
    DpiOutOfRange(u32),
    /// The rendered page would have too many pixels.
    #[error("{layout} would be {pixels} pixels, the limit is {max}")]
    TooManyPixels {
        /// The layout.
        layout: String,
        /// Pixels it would have.
        pixels: u64,
        /// The limit.
        max: u64,
    },
    /// Too many zoom levels.
    #[error("{layout} needs {levels} zoom levels, the limit is {max}")]
    TooManyZoomLevels {
        /// The layout.
        layout: String,
        /// Levels it needs.
        levels: u32,
        /// The limit.
        max: u32,
    },
    /// Too many tiles.
    #[error("{layout} needs {tiles} tiles, the limit is {max}")]
    TooManyTiles {
        /// The layout.
        layout: String,
        /// Tiles it needs.
        tiles: u64,
        /// The limit.
        max: u64,
    },
    /// The cancel token fired.
    #[error("cancelled")]
    Cancelled,
    /// The deadline passed.
    #[error("the deadline passed")]
    Deadline,
}

impl PmTilesPlotError {
    /// A stable code.
    #[must_use]
    pub fn code(&self) -> &'static str {
        "NOT_YET"
    }
}

/// Print layouts, plotted to one PMTiles archive each.
pub struct PmTilesPlot<'a> {
    #[allow(dead_code)]
    source: CadSource<'a>,
    #[allow(dead_code)]
    dest_dir: &'a Path,
}

impl<'a> PmTilesPlot<'a> {
    /// A plot of `source` into archives under `dest_dir`.
    pub fn new(source: CadSource<'a>, dest_dir: &'a Path) -> Self {
        Self { source, dest_dir }
    }
    /// The archive name stem.
    #[must_use]
    pub fn stem(self, _stem: impl Into<String>) -> Self {
        self
    }
    /// The view profile.
    #[must_use]
    pub fn view_profile(self, _profile: ViewProfile) -> Self {
        self
    }
    /// The drawing's unit.
    #[must_use]
    pub fn unit(self, _unit: SourceUnit) -> Self {
        self
    }
    /// The plot settings.
    #[must_use]
    pub fn plot_settings(self, _settings: PlotSettings) -> Self {
        self
    }
    /// The plot style table.
    #[must_use]
    pub fn plot_style_table(self, _table: PlotStyleTable) -> Self {
        self
    }
    /// The layouts.
    #[must_use]
    pub fn layouts<S: Into<String>>(self, _names: impl IntoIterator<Item = S>) -> Self {
        self
    }
    /// Raster resolution.
    #[must_use]
    pub fn dpi(self, _dpi: u32) -> Self {
        self
    }
    /// The bounds.
    #[must_use]
    pub fn limits(self, _limits: RasterLimits) -> Self {
        self
    }
    /// The cancel token.
    #[must_use]
    pub fn cancel(self, _cancel: CancelToken) -> Self {
        self
    }
    /// The observer.
    #[must_use]
    pub fn observer(self, _observer: Arc<dyn EngineObserver>) -> Self {
        self
    }
    /// The deadline.
    #[must_use]
    pub fn deadline(self, _within: Duration) -> Self {
        self
    }
    /// Writes the archives with provider `P`.
    ///
    /// # Errors
    ///
    /// [`PmTilesPlotError`].
    pub fn convert_with<P: DwgProvider>(self) -> Result<PmTilesPlotOutput, PmTilesPlotError> {
        Err(PmTilesPlotError::NotYet)
    }
}
