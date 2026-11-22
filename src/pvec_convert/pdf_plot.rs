//! Plot to PDF (#1636): one print layout written as one PDF page. This is the stub the red
//! tests in `tests/pvec_1636_pdf_plot.rs` compile against; nothing is written yet.

use std::path::{Path, PathBuf};

use acadrust::plot_style::PlotStyleTable;

use super::{PlotSettings, PvecConvertError};
use crate::cad::CadSource;
use crate::pvec::{DwgProvider, SourceUnit, ViewProfile};

/// What a PDF plot wrote.
#[derive(Debug)]
pub struct PdfPlotOutput {
    /// The PDF file.
    pub file: PathBuf,
    /// Its page count.
    pub pages: u32,
    /// What was not applied or was replaced, in the vocabulary of the sheet report's `notes`
    /// (`plot-paper-overridden`, ...). The PDF plot has no report section of its own.
    pub notes: Vec<String>,
}

/// A print layout, plotted to a PDF.
pub struct PdfPlot<'a> {
    #[allow(dead_code)]
    source: CadSource<'a>,
    #[allow(dead_code)]
    dest: &'a Path,
}

impl<'a> PdfPlot<'a> {
    /// A plot of `source` into the PDF `dest`.
    pub fn new(source: CadSource<'a>, dest: &'a Path) -> Self {
        Self { source, dest }
    }

    /// The layout to plot, as the print profile names it.
    #[must_use]
    pub fn view_profile(self, _profile: ViewProfile) -> Self {
        self
    }

    /// The drawing's unit.
    #[must_use]
    pub fn unit(self, _unit: SourceUnit) -> Self {
        self
    }

    /// The plot settings, the same type a PVec plot takes.
    #[must_use]
    pub fn plot_settings(self, _settings: PlotSettings) -> Self {
        self
    }

    /// The plot style table: stroke colour, width and screening per AutoCAD colour index.
    #[must_use]
    pub fn plot_style_table(self, _table: PlotStyleTable) -> Self {
        self
    }

    /// Only these paper layouts, one page each in tab order, matched the way
    /// [`PvecLayoutSet::layouts`](super::PvecLayoutSet::layouts) matches them. Without this the
    /// page is the layout the view profile names.
    #[must_use]
    pub fn layouts<S: Into<String>>(self, _names: impl IntoIterator<Item = S>) -> Self {
        self
    }

    /// Writes the PDF with provider `P`.
    ///
    /// # Errors
    ///
    /// [`PvecConvertError`] as a PVec conversion.
    pub fn convert_with<P: DwgProvider>(self) -> Result<PdfPlotOutput, PvecConvertError> {
        todo!("PDF plot, #1636")
    }
}
