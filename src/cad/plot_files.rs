//! One import path for the plot files a server reads and writes: PC3
//! plot configurations and CTB/STB plot style tables (issue #1634).
//!
//! Both kinds load through [`load_pc3`] and [`load_plot_style`] (issue #1673), which bound what
//! they read and refuse with a [`PlotFileError`] whose [`code`](PlotFileError::code) is stable.

use std::fmt;

pub use super::pc3::{
    MAX_PC3_BYTES, MAX_PC3_DEPTH, Pc3, Pc3Binary, Pc3Device, Pc3Error, Pc3File, Pc3Media,
};
#[cfg(feature = "dwg-acadrust")]
pub use acadrust::plot_style::{PlotStyleEntry, PlotStyleTable};

/// The most bytes a plot file may have before anything reads it: 1 MiB, the PC3 limit.
pub const MAX_PLOT_FILE_BYTES: usize = MAX_PC3_BYTES;

/// The most a CTB/STB may inflate to: 8 MiB. A real table is under 100 KB of text.
pub const MAX_PLOT_STYLE_INFLATED_BYTES: usize = 8 * 1024 * 1024;

/// Why a plot file was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum PlotFileError {
    /// The file is over [`MAX_PLOT_FILE_BYTES`].
    TooLarge(usize),
    /// A plot style table that inflates past [`MAX_PLOT_STYLE_INFLATED_BYTES`], or announces it.
    InflatedTooLarge,
    /// A PC3 whose JSON nests deeper than [`MAX_PC3_DEPTH`].
    TooDeep,
    /// Not a plot file this reader knows.
    Unrecognised,
    /// A plot file of a known kind that cannot be read, and why.
    Unreadable(String),
}

impl PlotFileError {
    /// A short code that stays the same from release to release, for a caller to match on.
    #[must_use]
    pub fn code(&self) -> &'static str {
        match self {
            PlotFileError::TooLarge(_) => "plot-file-too-large",
            PlotFileError::InflatedTooLarge => "plot-file-inflated-too-large",
            PlotFileError::TooDeep => "plot-file-too-deep",
            PlotFileError::Unrecognised => "plot-file-unrecognised",
            PlotFileError::Unreadable(_) => "plot-file-unreadable",
        }
    }
}

impl fmt::Display for PlotFileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PlotFileError::TooLarge(n) => {
                write!(
                    f,
                    "plot file is {n} bytes, over the {MAX_PLOT_FILE_BYTES}-byte limit"
                )
            }
            PlotFileError::InflatedTooLarge => write!(
                f,
                "plot style table inflates past the {MAX_PLOT_STYLE_INFLATED_BYTES}-byte limit"
            ),
            PlotFileError::TooDeep => write!(f, "PC3 nests deeper than {MAX_PC3_DEPTH} levels"),
            PlotFileError::Unrecognised => write!(f, "not a plot file"),
            PlotFileError::Unreadable(why) => write!(f, "plot file cannot be read: {why}"),
        }
    }
}

impl std::error::Error for PlotFileError {}

impl From<Pc3Error> for PlotFileError {
    fn from(error: Pc3Error) -> Self {
        match error {
            Pc3Error::TooLarge(n) => PlotFileError::TooLarge(n),
            Pc3Error::TooDeep => PlotFileError::TooDeep,
            Pc3Error::Unrecognised => PlotFileError::Unrecognised,
            other => PlotFileError::Unreadable(other.to_string()),
        }
    }
}

/// A PC3 of either kind, read from bytes the caller holds. The library never opens a file by a
/// name stored in a drawing.
pub fn load_pc3(bytes: &[u8]) -> Result<Pc3File, PlotFileError> {
    Ok(Pc3File::parse(bytes)?)
}

/// A CTB or STB read from bytes the caller holds, under the same bounds as a PC3.
#[cfg(feature = "dwg-acadrust")]
pub fn load_plot_style(name: &str, bytes: &[u8]) -> Result<PlotStyleTable, PlotFileError> {
    PlotStyleTable::from_bytes(name, bytes).map_err(|e| PlotFileError::Unreadable(e.to_string()))
}

/// What a PC3 says about the device that plots: the paper, the area of it the device prints, and
/// two settings of its PDF driver. A conversion takes them as gap-fillers under the layout's own
/// page setup (issue #1673).
#[derive(Debug, Clone, PartialEq)]
pub struct PlotDevice {
    /// The paper's width and height in millimetres.
    pub paper_mm: [f64; 2],
    /// The printable rectangle in millimetres: lower left x and y, upper right x and y.
    pub printable_mm: [f64; 4],
    /// The highest resolution a raster is kept at, in dpi (`Raster_Limit`). `None` when the PC3
    /// does not say.
    pub raster_limit_dpi: Option<f64>,
    /// Text stays text rather than outlines (`Font_Capture`). `None` when the PC3 does not say.
    pub text_as_text: Option<bool>,
}

impl PlotDevice {
    /// The device facts of a JSON PC3.
    pub fn from_pc3(pc3: &Pc3) -> Result<Self, Pc3Error> {
        let media = pc3.media()?;
        Ok(PlotDevice {
            paper_mm: media.size_mm,
            printable_mm: media.printable_mm,
            raster_limit_dpi: None,
            text_as_text: None,
        })
    }
}
