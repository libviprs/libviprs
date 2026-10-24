//! The PDF side of the vector boundary (N5.1).
//!
//! `pdf` only ever turned page geometry into pixels. This module reads the path
//! operators of a page's content stream instead, so a vector PDF reaches
//! normalization as geometry without being rasterized first. It is built on
//! `lopdf`, which is a non-optional dependency, so none of it needs the
//! `pdfium` feature.
//!
//! ```text
//! extract_page_paths(path | bytes, page, limits)
//!   tokenize /Contents (one stream or an array of them)
//!   m l c v y h re   ->  PdfPathSegment   (v and y expanded to full cubics)
//!   cm q Q           ->  CTM over a bounded stack
//!   n f f* F B B* b b* S s W W*  ->  PdfPaintIntent, recorded not interpreted
//!   Do (Form XObject) ->  /Matrix composed in, bounded depth and expansion
//!   page box origin removed, /Rotate composed  ->  PdfPathEntity
//! ```
//!
//! # Coordinates
//!
//! Every point in a [`PdfPathEntity`] is in page space: the CTM, the page box
//! origin and the page's `/Rotate` are already applied, the y axis still points
//! up as in PDF, and the origin is the lower-left corner of the page as it is
//! displayed. The numbers are `f64` widened from the `f32` `lopdf` parses, and
//! nothing here compares or flattens them: curve semantics are N5.2's and PVec's.
//!
//! # What crosses the boundary
//!
//! [`PdfPathEntity`] is shaped like the DWG side's `RawEntity`: a geometry
//! payload plus a provenance record. No `lopdf` object, dictionary or stream is
//! in any public signature, and `tests/pvec_pdf_vector.rs` enumerates the public
//! items and fails on one.
//!
//! # Limits
//!
//! [`PdfVectorLimits`] holds the bounds this adapter needs. N3.6 owns the shared
//! limit set and has not landed, so these are local and N3.6 can replace the
//! struct without touching the entity types. A limit that trips is a typed error
//! naming the limit and the observed value, never a truncated result.

use std::path::Path;

use thiserror::Error;

/// A page-space point, `[x, y]`.
pub type PdfPoint = [f64; 2];

/// Bounds on one extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PdfVectorLimits {
    /// Deepest Form XObject nesting. The page's own stream is depth 0.
    pub max_form_depth: u32,
    /// Most content-stream operations processed across the page and every form
    /// expansion, which bounds a form that is invoked many times.
    pub max_operations: u64,
    /// Deepest `q` nesting.
    pub max_graphics_stack: usize,
    /// Most entities returned.
    pub max_entities: usize,
}

impl Default for PdfVectorLimits {
    fn default() -> Self {
        Self {
            max_form_depth: 16,
            max_operations: 4_000_000,
            max_graphics_stack: 256,
            max_entities: 1_000_000,
        }
    }
}

/// One piece of a path, in page space.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum PdfPathSegment {
    /// Start a subpath.
    MoveTo(PdfPoint),
    /// A straight segment to the point.
    LineTo(PdfPoint),
    /// A cubic Bezier with both control points explicit. `v` and `y` arrive here
    /// already expanded.
    CurveTo {
        /// First control point.
        c1: PdfPoint,
        /// Second control point.
        c2: PdfPoint,
        /// End point.
        end: PdfPoint,
    },
    /// Close the current subpath.
    Close,
}

/// A fill or clip rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PdfFillRule {
    /// Nonzero winding.
    NonZero,
    /// Even-odd.
    EvenOdd,
}

/// What a painting operator asked for. Recorded, not interpreted: style is N5.4,
/// fills are N5.6.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PdfPaintIntent {
    /// The fill rule when the operator fills.
    pub fill: Option<PdfFillRule>,
    /// Whether the operator strokes.
    pub stroke: bool,
    /// Whether the operator closes the subpath first (`s`, `b`, `b*`).
    pub close: bool,
}

/// Where an entity came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PdfProvenance {
    /// The 1-based page number.
    pub page: usize,
    /// The painting operator that ended the path (`S`, `f*`, `n`).
    pub operator: String,
    /// The Form XObject names invoked on the way, outermost first. Empty for the
    /// page's own stream.
    pub form_path: Vec<String>,
    /// Position of this entity in the page's output order.
    pub sequence: u64,
}

/// One painted or clipped path.
#[derive(Debug, Clone, PartialEq)]
pub struct PdfPathEntity {
    /// The path, in page space.
    pub segments: Vec<PdfPathSegment>,
    /// What painting was asked for. Both parts are empty for `n`.
    pub paint: PdfPaintIntent,
    /// The rule of a `W` or `W*` that applied to this path.
    pub clip: Option<PdfFillRule>,
    /// Where it came from.
    pub provenance: PdfProvenance,
}

/// One page's paths.
#[derive(Debug, Clone, PartialEq)]
pub struct PdfVectorPage {
    /// The 1-based page number.
    pub page: usize,
    /// Displayed width after `/Rotate`, in points.
    pub width: f64,
    /// Displayed height after `/Rotate`, in points.
    pub height: f64,
    /// `/Rotate` in degrees, 0, 90, 180 or 270.
    pub rotation_degrees: i64,
    /// The entities in content-stream order, forms expanded in place.
    pub entities: Vec<PdfPathEntity>,
    /// `Q` operators that had no matching `q`, which are ignored.
    pub unbalanced_restores: u32,
}

/// Why an extraction failed.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum PdfVectorError {
    /// The file is missing, unreadable or not a PDF.
    #[error("PDF read error: {0}")]
    Read(String),
    /// The page number is outside the document.
    #[error("page {page} is out of range (document has {total})")]
    PageOutOfRange {
        /// The requested page.
        page: usize,
        /// Pages in the document.
        total: usize,
    },
    /// The document is encrypted, and this adapter reads no encrypted streams.
    #[error("the PDF is encrypted")]
    Encrypted,
    /// `/Rotate` is not a multiple of 90.
    #[error("unsupported /Rotate {0}")]
    UnsupportedRotation(i64),
    /// A content stream did not tokenize, or an operator was malformed. The page
    /// is not empty, it is unreadable.
    #[error("page {page}: content stream error: {detail}")]
    ContentStream {
        /// The page.
        page: usize,
        /// What went wrong.
        detail: String,
    },
    /// Form XObjects nested deeper than `max_form_depth`.
    #[error("form XObject depth limit max_form_depth={limit} exceeded at depth {observed}")]
    FormDepthExceeded {
        /// The configured limit.
        limit: u32,
        /// The depth reached.
        observed: u32,
    },
    /// More operations than `max_operations`.
    #[error("operation limit max_operations={limit} exceeded ({observed} operations)")]
    OperationLimitExceeded {
        /// The configured limit.
        limit: u64,
        /// The count reached.
        observed: u64,
    },
    /// `q` nested deeper than `max_graphics_stack`.
    #[error("graphics state stack limit max_graphics_stack={limit} exceeded at depth {observed}")]
    GraphicsStackExceeded {
        /// The configured limit.
        limit: usize,
        /// The depth reached.
        observed: usize,
    },
    /// More entities than `max_entities`.
    #[error("entity limit max_entities={limit} exceeded")]
    EntityLimitExceeded {
        /// The configured limit.
        limit: usize,
    },
}

/// Read the paths of one page of a PDF file. `page` is 1-based.
///
/// # Errors
///
/// See [`PdfVectorError`].
pub fn extract_page_paths(
    path: &Path,
    page: usize,
    limits: &PdfVectorLimits,
) -> Result<PdfVectorPage, PdfVectorError> {
    let bytes = std::fs::read(path).map_err(|e| PdfVectorError::Read(e.to_string()))?;
    extract_page_paths_from_bytes(&bytes, page, limits)
}

/// Read the paths of one page of a PDF held in memory. `page` is 1-based.
///
/// # Errors
///
/// See [`PdfVectorError`].
pub fn extract_page_paths_from_bytes(
    _bytes: &[u8],
    _page: usize,
    _limits: &PdfVectorLimits,
) -> Result<PdfVectorPage, PdfVectorError> {
    todo!("N5.1 implementation follows the red tests")
}
