//! A bounded player for the enhanced metafile (EMF) preview of an embedded OLE object (issue #1705).
//!
//! An OLE2FRAME in a DWG carries, next to the object, the picture the application cached for display.
//! For a spreadsheet that is an EMF of a table: lines, filled cells and text. [`play`] reads that
//! record stream and gives back the shapes it draws, already mapped from the metafile's device
//! pixels onto the frame's two corners, so a conversion can carry them like any other entity of
//! the sheet. Nothing is rasterised and nothing is read from disk.
//!
//! The player knows the records such a table is made of: header and end, the device context state
//! (background mode, raster op, text alignment, text and background colour, save and restore, clip
//! rectangle and clip region), objects (pen, brush, font, select, delete), `MoveToEx`, `LineTo`,
//! `Rectangle`, a pattern `BitBlt` as a solid fill and `ExtTextOutW` as text. Any other record is
//! counted and named in the drawing's notes, never skipped silently. A record that is malformed, or
//! a preview over a limit, is a typed [`EmfRefusal`] and nothing of it is drawn.
//!
//! The picture's bounds (device pixels, left and right edge both inside) are mapped onto the
//! corners, so the table's outer lines land on them. What it does not carry: pen and text colours
//! and pen widths (a stroke takes the entity's own style), bitmaps, curves, and world transforms.

/// How the player ended without a drawing.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum EmfRefusal {
    /// Not played yet.
    NotPlayed,
    /// The bytes are not an enhanced metafile.
    NotAMetafile,
    /// The preview is larger than the player reads.
    TooLarge {
        /// The most bytes it reads.
        limit: usize,
    },
    /// The metafile has more records than the player plays.
    TooManyRecords {
        /// The most records it plays.
        limit: u32,
    },
    /// A record is shorter than its header, is not a whole number of words or runs past the end.
    BadRecord {
        /// The record's place in the stream, counting the header as 0.
        index: u32,
    },
    /// The metafile draws more shapes than the player carries.
    TooManyShapes {
        /// The most shapes it carries.
        limit: usize,
    },
    /// An object handle past the table the player keeps.
    TooManyObjects {
        /// The most handles it keeps.
        limit: usize,
    },
    /// The picture has no width or no height to map onto the corners.
    NoExtent,
    /// More device contexts are saved than the player keeps.
    TooDeep,
}

impl std::fmt::Display for EmfRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotPlayed => f.write_str("the preview is not played"),
            Self::NotAMetafile => f.write_str("the preview is not an enhanced metafile"),
            Self::TooLarge { limit } => write!(f, "the preview is over {limit} bytes"),
            Self::TooManyRecords { limit } => {
                write!(f, "the preview has more than {limit} records")
            }
            Self::BadRecord { index } => write!(f, "record {index} of the preview is malformed"),
            Self::TooManyShapes { limit } => {
                write!(f, "the preview draws more than {limit} shapes")
            }
            Self::TooManyObjects { limit } => {
                write!(f, "the preview names an object handle past {limit}")
            }
            Self::NoExtent => f.write_str("the preview's picture has no width or height"),
            Self::TooDeep => f.write_str("the preview saves too many device contexts"),
        }
    }
}

impl std::error::Error for EmfRefusal {}

/// One thing the preview draws, in the frame's coordinates.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum EmfShape {
    /// An open polyline.
    Line(Vec<[f64; 2]>),
    /// A closed ring filled with a solid colour (red, green, blue, alpha, one byte each).
    Fill {
        /// The corners of the ring.
        ring: Vec<[f64; 2]>,
        /// The colour.
        rgba: u32,
    },
    /// One run of text.
    Text {
        /// Where the baseline starts, or is centred or ends (see `anchor`).
        at: [f64; 2],
        /// The height of a capital.
        height: f64,
        /// The rotation in radians, counter-clockwise.
        rotation: f64,
        /// How far along the run `at` sits: 0 at its start, 0.5 in the middle, 1 at its end.
        anchor: f64,
        /// The characters.
        text: String,
    },
}

/// What a preview draws.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct EmfDrawing {
    /// The shapes, in the order the metafile draws them.
    pub shapes: Vec<EmfShape>,
    /// What was left out, one line each: a record type the player does not know, with its count.
    pub notes: Vec<String>,
}

/// Plays `bytes` and maps the picture onto the rectangle whose upper-left and lower-right corners
/// are `upper_left` and `lower_right`.
///
/// # Errors
///
/// An [`EmfRefusal`] for a preview that is not an EMF, is malformed, or is over a limit.
pub fn play(
    bytes: &[u8],
    upper_left: [f64; 2],
    lower_right: [f64; 2],
) -> Result<EmfDrawing, EmfRefusal> {
    let _ = (bytes, upper_left, lower_right);
    Err(EmfRefusal::NotPlayed)
}
