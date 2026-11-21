//! PC3 plot configuration files (issue #1634).

use std::fmt;

/// Largest PC3 accepted, in bytes.
pub const MAX_PC3_BYTES: usize = 1024 * 1024;
/// Deepest JSON nesting accepted.
pub const MAX_PC3_DEPTH: usize = 32;

/// Why a PC3 was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pc3Error {
    /// The file is over [`MAX_PC3_BYTES`].
    TooLarge(usize),
    /// The JSON nests deeper than [`MAX_PC3_DEPTH`].
    TooDeep,
    /// Not a PC3 this reader knows.
    Unrecognised,
    /// The JSON does not parse.
    Syntax(usize, &'static str),
    /// A binary PC3 is kept verbatim and cannot be edited.
    BinaryNotEditable(&'static str),
    /// The document lacks a section a typed accessor needs.
    Missing(&'static str),
}

impl fmt::Display for Pc3Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for Pc3Error {}

/// A PC3 of either kind.
#[derive(Debug, Clone)]
pub enum Pc3File {
    /// The JSON kind, which can be read and edited.
    Json(Pc3),
    /// A binary kind, stored as it came.
    Binary(Pc3Binary),
}

/// A binary PC3 (`PC3VER1` or `PIAFILEVERSION_1.0`), held verbatim.
#[derive(Debug, Clone)]
pub struct Pc3Binary {
    /// The format's name.
    pub format: &'static str,
    /// The file as read.
    pub bytes: Vec<u8>,
}

/// A JSON PC3.
#[derive(Debug, Clone)]
pub struct Pc3 {}

/// The paper half of a PC3, in millimetres.
#[derive(Debug, Clone, PartialEq)]
pub struct Pc3Media {
    /// The paper's name, e.g. `ANSI_B_(11.00_x_17.00_Inches)`.
    pub paper_name: String,
    /// Media width and height.
    pub size_mm: [f64; 2],
    /// Printable rectangle: lower left x and y, upper right x and y.
    pub printable_mm: [f64; 4],
    /// Landscape mode of the paper size.
    pub landscape: bool,
    /// The device's default orientation (`dm_orientation`).
    pub orientation: i64,
}

/// Which device a PC3 names. Never rewritten.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pc3Device {
    /// `localized_family_name`.
    pub family: String,
    /// `localized_model_name`.
    pub model: String,
}

impl Pc3File {
    /// Read a PC3 of either kind.
    pub fn parse(_bytes: &[u8]) -> Result<Self, Pc3Error> {
        todo!()
    }
    /// The file's bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        todo!()
    }
    /// The editable document, or the reason there is none.
    pub fn editable(&mut self) -> Result<&mut Pc3, Pc3Error> {
        todo!()
    }
}

impl Pc3 {
    /// Read a JSON PC3.
    pub fn parse(_bytes: &[u8]) -> Result<Self, Pc3Error> {
        todo!()
    }
    /// The file's bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        todo!()
    }
    /// The raw text of a scalar at `path`, as the file spells it.
    pub fn scalar(&self, _path: &[&str]) -> Option<&str> {
        todo!()
    }
    /// The paper half.
    pub fn media(&self) -> Result<Pc3Media, Pc3Error> {
        todo!()
    }
    /// Change the paper half; values that did not change keep their text.
    pub fn set_media(&mut self, _media: &Pc3Media) -> Result<(), Pc3Error> {
        todo!()
    }
    /// The device the file names.
    pub fn device(&self) -> Option<Pc3Device> {
        todo!()
    }
}
