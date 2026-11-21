//! Plot to PDF (#1636): one print layout written as one PDF page. This is the stub the red
//! tests in `tests/pvec_1636_pdf_plot.rs` compile against; nothing is written yet.

use std::path::{Path, PathBuf};

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

    /// Plot style for one AutoCAD colour index: stroke colour and width in millimetres. A
    /// placeholder for the plot style table type that #1630 brings.
    #[must_use]
    pub fn style_for_aci(self, _aci: u8, _rgb: [u8; 3], _width_mm: f64) -> Self {
        self
    }

    /// One page per paper layout, in tab order, instead of the profile's single layout.
    #[must_use]
    pub fn all_layouts(self) -> Self {
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
