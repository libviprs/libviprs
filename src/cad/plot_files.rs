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
///
/// The size is checked first: a binary PC3 is kept verbatim, so `Pc3File::parse` alone would copy
/// a file of any size.
pub fn load_pc3(bytes: &[u8]) -> Result<Pc3File, PlotFileError> {
    if bytes.len() > MAX_PLOT_FILE_BYTES {
        return Err(PlotFileError::TooLarge(bytes.len()));
    }
    Ok(Pc3File::parse(bytes)?)
}

/// A CTB or STB read from bytes the caller holds. The file is held to [`MAX_PLOT_FILE_BYTES`], and
/// what it inflates to, or says it will, to [`MAX_PLOT_STYLE_INFLATED_BYTES`]. The reader's own
/// ceiling is 64 MiB, which is a lot of memory for a table of a few hundred lines.
#[cfg(feature = "dwg-acadrust")]
pub fn load_plot_style(name: &str, bytes: &[u8]) -> Result<PlotStyleTable, PlotFileError> {
    if bytes.len() > MAX_PLOT_FILE_BYTES {
        return Err(PlotFileError::TooLarge(bytes.len()));
    }
    if !bytes.starts_with(b"PIAFILEVERSION_2.0") {
        return Err(PlotFileError::Unrecognised);
    }
    inflation_within_bound(bytes)?;
    PlotStyleTable::from_bytes(name, bytes).map_err(|e| PlotFileError::Unreadable(e.to_string()))
}

/// Refuses a CTB/STB that announces or inflates past [`MAX_PLOT_STYLE_INFLATED_BYTES`], reading at
/// most one byte past it and keeping none. The reader falls back to raw deflate when a zlib
/// trailer is wrong, so both are counted.
#[cfg(feature = "dwg-acadrust")]
fn inflation_within_bound(bytes: &[u8]) -> Result<(), PlotFileError> {
    use flate2::read::{DeflateDecoder, ZlibDecoder};
    use std::io::{Read, copy, sink};
    const PREFIX: &[u8] = b"PIAFILEVERSION_2.0,CTBVER1,compress\r\npmzlibcodec";
    let bound = MAX_PLOT_STYLE_INFLATED_BYTES as u64;
    let past =
        |r: &mut dyn Read| copy(&mut r.take(bound + 1), &mut sink()).is_ok_and(|n| n > bound);
    let payload = if bytes.starts_with(PREFIX) {
        let word = |at: usize| {
            bytes
                .get(at..at + 4)
                .map(|w| u32::from_le_bytes(w.try_into().unwrap()))
        };
        let (Some(announced), Some(packed)) = (word(52), word(56)) else {
            return Ok(()); // a cut header: the reader says so
        };
        if u64::from(announced) > bound {
            return Err(PlotFileError::InflatedTooLarge);
        }
        let end = 60usize.saturating_add(packed as usize).min(bytes.len());
        bytes.get(60..end).unwrap_or_default()
    } else {
        let start = bytes.iter().position(|&b| b == b'\n').map_or(0, |p| p + 1);
        &bytes[start..]
    };
    let raw_after_header = payload.get(2..).unwrap_or_default();
    if past(&mut ZlibDecoder::new(payload))
        || past(&mut DeflateDecoder::new(raw_after_header))
        || past(&mut DeflateDecoder::new(payload))
    {
        return Err(PlotFileError::InflatedTooLarge);
    }
    Ok(())
}

/// The built-in monochrome plot style table: all 255 colours plot black at 100 percent screening and
/// lineweight 9 (0.25 mm), the table a plot with no table of its own is drawn with (issue #1670).
#[cfg(feature = "dwg-acadrust")]
#[must_use]
pub fn default_plot_style() -> PlotStyleTable {
    todo!("issue #1670")
}

/// The built-in PDF PC3, the JSON kind: the PDF device, text kept as text, a raster limit of 400 dpi
/// and a full bleed ISO A3 page, which only fills a layout that has no paper of its own (issue #1670).
#[must_use]
pub fn default_pc3() -> Pc3 {
    todo!("issue #1670")
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
            raster_limit_dpi: pc3
                .custom("Raster_Limit")
                .and_then(|v| v.parse::<f64>().ok())
                .filter(|v| v.is_finite() && *v > 0.0),
            text_as_text: match pc3.custom("Font_Capture") {
                Some("true") => Some(true),
                Some("false") => Some(false),
                _ => None,
            },
        })
    }
}
