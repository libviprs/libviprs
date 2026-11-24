//! Plot to PMTiles (#1676): one raster pyramid archive per plotted sheet.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use acadrust::plot_style::PlotStyleTable;

use super::{PdfPlot, PlotSettings, PvecConvertError};
use crate::cad::CadSource;
use crate::observe::EngineObserver;
use crate::pdf::PageSizing;
use crate::planner::{Layout, PlannerError, PyramidPlan, PyramidPlanner};
use crate::pvec::view_profile::{SpaceSelection, ViewProfileError};
use crate::pvec::{DwgProvider, LayoutKind, SourceUnit, ViewProfile};
use crate::sink::SinkError;
use crate::sink_pmtiles::PmTilesSink;
use crate::{CancelToken, EngineBuilder, EngineError, PdfError};

/// The tile edge, in pixels.
const TILE: u32 = 256;
/// The dpi range a plot may be rendered at.
const DPI_RANGE: std::ops::RangeInclusive<u32> = 36..=600;
/// PMTiles stores the zoom in five bits.
const ZOOM_LEVELS_MAX: u32 = 32;

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
    /// The page could not be rendered.
    #[error("the page could not be rendered")]
    Render(#[source] PdfError),
    /// The pyramid could not be planned.
    #[error("the pyramid could not be planned")]
    Pyramid(#[source] PlannerError),
    /// The tiles could not be written.
    #[error("the archive could not be written")]
    Engine(#[source] EngineError),
    /// The archive sink could not be built.
    #[error("the archive could not be opened")]
    Sink(#[source] SinkError),
    /// A file or directory could not be made.
    #[error("the output could not be written")]
    Io(#[source] std::io::Error),
    /// The file stem is not a plain file name.
    #[error("the file stem {0:?} is not a plain file name")]
    BadStem(String),
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
        match self {
            Self::Plot(e) => e.code(),
            Self::PageTooLarge { .. } => "PLOT_PAGE_TOO_LARGE",
            Self::DpiOutOfRange(_) => "PLOT_DPI_OUT_OF_RANGE",
            Self::TooManyPixels { .. } => "PLOT_PIXELS",
            Self::TooManyZoomLevels { .. } => "PLOT_ZOOM_LEVELS",
            Self::TooManyTiles { .. } => "PLOT_TILE_COUNT",
            Self::Render(_) => "RENDER_FAILED",
            Self::Pyramid(_) => "PYRAMID_FAILED",
            Self::Engine(_) => "TILES_FAILED",
            Self::Sink(_) => "ARCHIVE_FAILED",
            Self::Io(_) => "PUBLISH_FAILED",
            Self::BadStem(_) => "PLOT_STEM",
            Self::Cancelled => "CANCELLED",
            Self::Deadline => "DEADLINE",
        }
    }
}

/// Print layouts, plotted to one PMTiles archive each.
///
/// Each layout is plotted by [`PdfPlot`] into a scratch PDF, its page is rendered with pdfium at
/// `dpi`, and that raster is cut into an XYZ pyramid of PNG tiles written through
/// [`PmTilesSink`]. The archives are `<stem>.<NN>-<slug>.pmtiles`, numbered in tab order, with
/// the slug the layout name with every byte outside `[a-z0-9]` folded to `-`. Every bound in
/// [`RasterLimits`] is checked on all the pages before the first is rendered, so a refusal writes
/// nothing, and a plot that fails part way removes the archives it had written.
pub struct PmTilesPlot<'a> {
    source: CadSource<'a>,
    dest_dir: &'a Path,
    stem: String,
    profile: ViewProfile,
    unit: Option<SourceUnit>,
    settings: PlotSettings,
    table: Option<PlotStyleTable>,
    layouts: Option<Vec<String>>,
    dpi: u32,
    limits: RasterLimits,
    cancel: CancelToken,
    observer: Option<Arc<dyn EngineObserver>>,
    deadline: Option<Duration>,
}

impl<'a> PmTilesPlot<'a> {
    /// A plot of `source` into archives under `dest_dir`, at 150 dpi.
    pub fn new(source: CadSource<'a>, dest_dir: &'a Path) -> Self {
        Self {
            source,
            dest_dir,
            stem: "plot".to_owned(),
            profile: ViewProfile::default_profile(),
            unit: None,
            settings: PlotSettings::default(),
            table: None,
            layouts: None,
            dpi: 150,
            limits: RasterLimits::default(),
            cancel: CancelToken::new(),
            observer: None,
            deadline: None,
        }
    }

    /// The archive name stem (`plot` by default).
    #[must_use]
    pub fn stem(mut self, stem: impl Into<String>) -> Self {
        self.stem = stem.into();
        self
    }

    /// The layout to plot, as the print profile names it.
    #[must_use]
    pub fn view_profile(mut self, profile: ViewProfile) -> Self {
        self.profile = profile;
        self
    }

    /// The drawing's unit, handed to [`PdfPlot`] as it is.
    #[must_use]
    pub fn unit(mut self, unit: SourceUnit) -> Self {
        self.unit = Some(unit);
        self
    }

    /// The plot settings, as [`PdfPlot::plot_settings`].
    #[must_use]
    pub fn plot_settings(mut self, settings: PlotSettings) -> Self {
        self.settings = settings;
        self
    }

    /// The plot style table, as [`PdfPlot::plot_style_table`].
    #[must_use]
    pub fn plot_style_table(mut self, table: PlotStyleTable) -> Self {
        self.table = Some(table);
        self
    }

    /// Only these paper layouts, one archive each in tab order, matched as [`PdfPlot::layouts`].
    #[must_use]
    pub fn layouts<S: Into<String>>(mut self, names: impl IntoIterator<Item = S>) -> Self {
        self.layouts = Some(names.into_iter().map(Into::into).collect());
        self
    }

    /// The resolution the page is rendered at, 36 to 600 (150 by default).
    #[must_use]
    pub fn dpi(mut self, dpi: u32) -> Self {
        self.dpi = dpi;
        self
    }

    /// The bounds on a sheet's size.
    #[must_use]
    pub fn limits(mut self, limits: RasterLimits) -> Self {
        self.limits = limits;
        self
    }

    /// The cancel token, polled before each page and inside its plot and its tiling.
    #[must_use]
    pub fn cancel(mut self, cancel: CancelToken) -> Self {
        self.cancel = cancel;
        self
    }

    /// Receives the plot's events, as [`PdfPlot::observer`].
    #[must_use]
    pub fn observer(mut self, observer: Arc<dyn EngineObserver>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// A wall-clock bound on the whole plot, counted from `convert_with`.
    #[must_use]
    pub fn deadline(mut self, within: Duration) -> Self {
        self.deadline = Some(within);
        self
    }

    /// Writes the archives with provider `P`.
    ///
    /// # Errors
    ///
    /// [`PmTilesPlotError`]: a bound in [`RasterLimits`] or the dpi range is
    /// [`PageTooLarge`](PmTilesPlotError::PageTooLarge) and its siblings, an unknown layout or a
    /// failed plot is [`Plot`](PmTilesPlotError::Plot).
    pub fn convert_with<P: DwgProvider>(self) -> Result<PmTilesPlotOutput, PmTilesPlotError> {
        if !DPI_RANGE.contains(&self.dpi) {
            return Err(PmTilesPlotError::DpiOutOfRange(self.dpi));
        }
        let cancel = match self.deadline.and_then(|d| Instant::now().checked_add(d)) {
            Some(at) => self.cancel.with_deadline(at),
            None => self.cancel.clone(),
        };
        if let Some(e) = stopped(&cancel) {
            return Err(e);
        }
        let wanted = self.wanted_layouts::<P>()?;

        fs::create_dir_all(self.dest_dir).map_err(PmTilesPlotError::Io)?;
        let work = self.dest_dir.join(format!("{}.plot-work", self.stem));
        fs::create_dir_all(&work).map_err(PmTilesPlotError::Io)?;
        let mut written: Vec<PathBuf> = Vec::new();
        let result = self.write_sheets::<P>(&wanted, &work, &cancel, &mut written);
        // The scratch directory is this call's own, so it goes whether or not it worked.
        let _ = fs::remove_dir_all(&work);
        if result.is_err() {
            for file in &written {
                let _ = fs::remove_file(file);
            }
        }
        result
    }

    fn write_sheets<P: DwgProvider>(
        &self,
        wanted: &[String],
        work: &Path,
        cancel: &CancelToken,
        written: &mut Vec<PathBuf>,
    ) -> Result<PmTilesPlotOutput, PmTilesPlotError> {
        let pdf = work.join("plot.pdf");
        let mut plot = PdfPlot::new(self.source, &pdf)
            .view_profile(self.profile.clone())
            .plot_settings(self.settings.clone())
            .layouts(wanted.iter().cloned())
            .cancel(cancel.clone());
        if let Some(unit) = self.unit {
            plot = plot.unit(unit);
        }
        if let Some(table) = &self.table {
            plot = plot.plot_style_table(table.clone());
        }
        if let Some(observer) = &self.observer {
            plot = plot.observer(observer.clone());
        }
        let plotted = plot.convert_with::<P>().map_err(|e| match e {
            PvecConvertError::Cancelled => PmTilesPlotError::Cancelled,
            PvecConvertError::Deadline => PmTilesPlotError::Deadline,
            other => PmTilesPlotError::Plot(other),
        })?;
        let info = crate::pdf_info(&pdf).map_err(PmTilesPlotError::Render)?;
        if info.pages.len() != wanted.len() {
            return Err(PmTilesPlotError::Io(std::io::Error::other(format!(
                "the plot has {} pages for {} layouts",
                info.pages.len(),
                wanted.len()
            ))));
        }
        // Every page is checked before the first is rendered, so a refusal writes nothing.
        for (name, page) in wanted.iter().zip(&info.pages) {
            self.refuse_page(name, page.width_pts, page.height_pts)?;
        }

        let mut sheets = Vec::with_capacity(wanted.len());
        for (n, name) in wanted.iter().enumerate() {
            if let Some(e) = stopped(cancel) {
                return Err(e);
            }
            let raster = crate::render_page_pdfium(&pdf, n + 1, self.dpi)
                .map_err(PmTilesPlotError::Render)?;
            let plan = self.plan_pixels(name, raster.width(), raster.height())?;
            let slug: String = name
                .bytes()
                .map(|b| {
                    if b.is_ascii_alphanumeric() {
                        char::from(b.to_ascii_lowercase())
                    } else {
                        '-'
                    }
                })
                .collect();
            let file = self
                .dest_dir
                .join(format!("{}.{:02}-{slug}.pmtiles", self.stem, n + 1));
            written.push(file.clone());
            let sink = PmTilesSink::builder(&file)
                .plan(plan.clone())
                .build()
                .map_err(PmTilesPlotError::Sink)?;
            EngineBuilder::new(&raster, plan.clone(), sink)
                .with_cancel(cancel.clone())
                .run()
                .map_err(|e| match e {
                    EngineError::Cancelled => {
                        stopped(cancel).unwrap_or(PmTilesPlotError::Cancelled)
                    }
                    other => PmTilesPlotError::Engine(other),
                })?;
            sheets.push(PmTilesSheet {
                layout: name.clone(),
                file,
                width_px: raster.width(),
                height_px: raster.height(),
                min_zoom: 0,
                max_zoom: (plan.levels.len() - 1) as u8,
                tiles: plan.total_tile_count(),
            });
        }
        Ok(PmTilesPlotOutput {
            sheets,
            notes: plotted.notes,
        })
    }

    /// Refuses a page of `w_pts` by `h_pts` that is past a bound, before it is rendered.
    fn refuse_page(&self, layout: &str, w_pts: f64, h_pts: f64) -> Result<(), PmTilesPlotError> {
        let long = w_pts.max(h_pts);
        if !long.is_finite() || long > self.limits.max_page_pts {
            return Err(PmTilesPlotError::PageTooLarge {
                layout: layout.to_owned(),
                long_pts: long,
                max_pts: self.limits.max_page_pts,
            });
        }
        let (w, h) = PageSizing::Exact.pixel_dims(w_pts, h_pts, self.dpi);
        self.plan_pixels(layout, w, h).map(|_| ())
    }

    /// The pyramid of a raster of `w` by `h` pixels, or the bound it is past.
    fn plan_pixels(&self, layout: &str, w: u32, h: u32) -> Result<PyramidPlan, PmTilesPlotError> {
        let pixels = u64::from(w) * u64::from(h);
        if pixels > self.limits.max_pixels {
            return Err(PmTilesPlotError::TooManyPixels {
                layout: layout.to_owned(),
                pixels,
                max: self.limits.max_pixels,
            });
        }
        let plan = PyramidPlanner::new(w, h, TILE, 0, Layout::Xyz)
            .map_err(PmTilesPlotError::Pyramid)?
            .plan();
        let levels = plan.levels.len() as u32;
        let max = self.limits.max_zoom_levels.min(ZOOM_LEVELS_MAX);
        if levels > max {
            return Err(PmTilesPlotError::TooManyZoomLevels {
                layout: layout.to_owned(),
                levels,
                max,
            });
        }
        let tiles = plan.total_tile_count();
        if tiles > self.limits.max_tiles {
            return Err(PmTilesPlotError::TooManyTiles {
                layout: layout.to_owned(),
                tiles,
                max: self.limits.max_tiles,
            });
        }
        Ok(plan)
    }

    /// The paper layouts to plot, in tab order, named as the drawing names them.
    fn wanted_layouts<P: DwgProvider>(&self) -> Result<Vec<String>, PmTilesPlotError> {
        let plot = PmTilesPlotError::Plot;
        let provider = P::open(self.source).map_err(|e| plot(PvecConvertError::Open(e)))?;
        let paper: Vec<String> = provider
            .layouts()
            .map_err(|e| plot(PvecConvertError::View(e.into())))?
            .into_iter()
            .filter(|l| l.kind == LayoutKind::Paper)
            .map(|l| l.name)
            .collect();
        let not_found = |name: &str| {
            plot(PvecConvertError::View(ViewProfileError::LayoutNotFound {
                requested: name.to_owned(),
                available: paper.clone(),
            }))
        };
        let find = |name: &str| {
            paper
                .iter()
                .position(|p| p.eq_ignore_ascii_case(name))
                .ok_or_else(|| not_found(name))
        };
        let mut at = match (&self.layouts, &self.profile.space) {
            (Some(names), _) => names
                .iter()
                .map(|n| find(n))
                .collect::<Result<Vec<_>, _>>()?,
            (None, SpaceSelection::PaperLayout(name)) => vec![find(name)?],
            (None, _) => return Err(not_found("")),
        };
        at.sort_unstable();
        at.dedup();
        Ok(at.into_iter().map(|i| paper[i].clone()).collect())
    }
}

/// The error a stopped token stands for: the deadline when it is what fired, else the cancel.
fn stopped(cancel: &CancelToken) -> Option<PmTilesPlotError> {
    if cancel.deadline_passed() {
        Some(PmTilesPlotError::Deadline)
    } else if cancel.is_cancelled() {
        Some(PmTilesPlotError::Cancelled)
    } else {
        None
    }
}
