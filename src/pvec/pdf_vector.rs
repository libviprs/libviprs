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

use lopdf::content::{Content, Operation};
use lopdf::{Dictionary, Document, Object};
use thiserror::Error;

use crate::pdf::{PageRotation, PdfError, page_object_id, page_rotation_of, page_visible_box};

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
    /// Most decoded content-stream bytes across the page and every form expansion.
    /// A compressed stream is cut off while it decodes, so the bound is on memory
    /// and not only on what is eventually read.
    pub max_content_bytes: u64,
    /// Most segments in one path, counted from the path's first operator to the
    /// operator that paints it.
    pub max_segments_per_path: usize,
}

impl Default for PdfVectorLimits {
    fn default() -> Self {
        Self {
            max_form_depth: 16,
            max_operations: 4_000_000,
            max_graphics_stack: 256,
            max_entities: 1_000_000,
            max_content_bytes: 64 << 20,
            max_segments_per_path: 1_000_000,
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
    /// The `sequence` of every clip-defining entity in force when this path was painted,
    /// outermost first. A path's own `W` takes effect after it is painted, so it is not in
    /// its own list. Scoped by `q`/`Q` and by Form XObject boundaries (N5.3).
    pub active_clips: Vec<u64>,
    /// Whether a soft mask (`/SMask` set through `gs`) was in the graphics state. The mask is
    /// recorded, not evaluated (N5.3).
    pub soft_masked: bool,
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
    /// `sh` operators seen, counted and not interpreted (N5.6).
    pub shading_paints: u32,
    /// Colour operators (`scn`, `SCN`) that selected a named pattern, counted and not
    /// interpreted (N5.6).
    pub pattern_selections: u32,
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
    /// More decoded content-stream bytes than `max_content_bytes`.
    #[error("content stream limit max_content_bytes={limit} exceeded ({observed} bytes)")]
    ContentBytesExceeded {
        /// The configured limit.
        limit: u64,
        /// The count reached when decoding stopped.
        observed: u64,
    },
    /// One path with more segments than `max_segments_per_path`.
    #[error("path segment limit max_segments_per_path={limit} exceeded ({observed} segments)")]
    PathSegmentsExceeded {
        /// The configured limit.
        limit: usize,
        /// The count reached.
        observed: usize,
    },
    /// A `/Resources` lookup came back to an object it had already visited.
    #[error("/Resources reference cycle through object {object}")]
    ResourceCycle {
        /// The object number met twice.
        object: u32,
    },
    /// The page box has no usable extent.
    #[error("page {page}: unusable page box: {detail}")]
    InvalidPageBox {
        /// The page.
        page: usize,
        /// What is wrong with it.
        detail: String,
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
    bytes: &[u8],
    page: usize,
    limits: &PdfVectorLimits,
) -> Result<PdfVectorPage, PdfVectorError> {
    let doc = Document::load_mem(bytes).map_err(|e| PdfVectorError::Read(e.to_string()))?;
    if doc.is_encrypted() {
        return Err(PdfVectorError::Encrypted);
    }
    let pages = doc.get_pages();
    let page_id = page_object_id(&pages, page).map_err(|e| match e {
        PdfError::PageOutOfRange { page, total } => PdfVectorError::PageOutOfRange { page, total },
        other => PdfVectorError::Read(other.to_string()),
    })?;

    let rotation = page_rotation_of(&doc, page_id).map_err(|e| match e {
        PdfError::UnsupportedRotation(d) => PdfVectorError::UnsupportedRotation(d),
        other => PdfVectorError::Read(other.to_string()),
    })?;
    // No /MediaBox at all: the spec's default, US Letter.
    let [l, b, r, t] = page_visible_box(&doc, page_id).unwrap_or([0.0, 0.0, 612.0, 792.0]);
    let (w, h) = (r - l, t - b);
    if ![l, b, r, t, w, h].iter().all(|v| v.is_finite()) || w <= 0.0 || h <= 0.0 {
        return Err(PdfVectorError::InvalidPageBox {
            page,
            detail: format!("[{l} {b} {r} {t}] has no finite, positive extent"),
        });
    }
    let base = Matrix::translate(-l, -b).then(&rotation_matrix(rotation, w, h));
    let (width, height) = match rotation {
        PageRotation::Quarter | PageRotation::ThreeQuarter => (h, w),
        _ => (w, h),
    };

    let mut content_bytes = 0u64;
    let content = page_content(&doc, page_id, page, limits, &mut content_bytes)?;
    let ops = decode(&content, page, limits, 0, 0)?;
    drop(content);
    let mut run = Run {
        content_bytes,
        doc: &doc,
        limits,
        page,
        out: Vec::new(),
        operations: 0,
        unbalanced: 0,
        shadings: 0,
        patterns: 0,
        form_path: Vec::new(),
        stack: Vec::new(),
        clips: Vec::new(),
        soft_mask: false,
    };
    let resources = inherited_resources(&doc, page_id)?;
    run.stream(&ops, resources, base, 0)?;

    Ok(PdfVectorPage {
        page,
        width,
        height,
        rotation_degrees: rotation.as_degrees(),
        entities: run.out,
        unbalanced_restores: run.unbalanced,
        shading_paints: run.shadings,
        pattern_selections: run.patterns,
    })
}

/// `[a b c d e f]` in PDF's row-vector convention: `x' = a x + c y + e`,
/// `y' = b x + d y + f`.
#[derive(Debug, Clone, Copy)]
struct Matrix([f64; 6]);

impl Matrix {
    const IDENTITY: Self = Self([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);

    fn translate(x: f64, y: f64) -> Self {
        Self([1.0, 0.0, 0.0, 1.0, x, y])
    }

    /// `self` applied first, then `next`.
    fn then(&self, next: &Self) -> Self {
        let [a, b, c, d, e, f] = self.0;
        let [na, nb, nc, nd, ne, nf] = next.0;
        Self([
            a * na + b * nc,
            a * nb + b * nd,
            c * na + d * nc,
            c * nb + d * nd,
            e * na + f * nc + ne,
            e * nb + f * nd + nf,
        ])
    }

    fn apply(&self, p: PdfPoint) -> PdfPoint {
        let [a, b, c, d, e, f] = self.0;
        [a * p[0] + c * p[1] + e, b * p[0] + d * p[1] + f]
    }
}

/// `/Rotate` as an explicit matrix over a `w` by `h` page-relative box, so the
/// result has the displayed page's lower-left as origin and y still up. The
/// rotation is clockwise, as `/Rotate` is.
fn rotation_matrix(rotation: PageRotation, w: f64, h: f64) -> Matrix {
    match rotation {
        PageRotation::Zero => Matrix::IDENTITY,
        PageRotation::Quarter => Matrix([0.0, -1.0, 1.0, 0.0, 0.0, w]),
        PageRotation::Half => Matrix([-1.0, 0.0, 0.0, -1.0, w, h]),
        PageRotation::ThreeQuarter => Matrix([0.0, 1.0, -1.0, 0.0, h, 0.0]),
    }
}

fn stream_error(page: usize, detail: impl Into<String>) -> PdfVectorError {
    PdfVectorError::ContentStream {
        page,
        detail: detail.into(),
    }
}

/// The text of one content stream, decoded when it has a filter and as stored when it
/// has none. `used` is the running total for the page, and decoding stops as soon as
/// it would pass `limits.max_content_bytes`, so a stream that inflates to gigabytes is
/// never held. A filter that fails is an error, not a fall back to the raw bytes.
///
/// Only `FlateDecode` is decoded here, either alone or as the last stage after
/// `ASCIIHexDecode`/`ASCII85Decode` (which can only shrink). Another filter would be
/// an expansion this function cannot bound, so it is refused, not guessed at.
fn stream_bytes(
    stream: &lopdf::Stream,
    page: usize,
    limits: &PdfVectorLimits,
    used: &mut u64,
) -> Result<Vec<u8>, PdfVectorError> {
    let over = |observed: u64| PdfVectorError::ContentBytesExceeded {
        limit: limits.max_content_bytes,
        observed,
    };
    let remaining = limits.max_content_bytes.saturating_sub(*used);
    let filters: Vec<Vec<u8>> = match stream.dict.get(b"Filter") {
        Err(_) => Vec::new(),
        Ok(Object::Name(n)) => vec![n.clone()],
        Ok(Object::Array(a)) => a
            .iter()
            .map(|o| {
                o.as_name()
                    .map(<[u8]>::to_vec)
                    .map_err(|_| stream_error(page, "stream filter: entry is not a name"))
            })
            .collect::<Result<_, _>>()?,
        Ok(_) => {
            return Err(stream_error(
                page,
                "stream filter: /Filter is not a name or array",
            ));
        }
    };
    let out = if filters.is_empty() {
        *used = used.saturating_add(stream.content.len() as u64);
        if stream.content.len() as u64 > remaining {
            return Err(over(*used));
        }
        stream.content.clone()
    } else {
        let (last, head) = filters.split_last().expect("not empty");
        let ascii = |f: &Vec<u8>| {
            matches!(
                f.as_slice(),
                b"ASCIIHexDecode" | b"AHx" | b"ASCII85Decode" | b"A85"
            )
        };
        let flate = matches!(last.as_slice(), b"FlateDecode" | b"Fl");
        if !head.iter().all(ascii) || !(flate || ascii(last)) {
            return Err(stream_error(
                page,
                format!(
                    "stream filter: unsupported chain {:?}",
                    filters
                        .iter()
                        .map(|f| String::from_utf8_lossy(f).into_owned())
                        .collect::<Vec<_>>()
                ),
            ));
        }
        if stream.dict.get(b"DecodeParms").is_ok() || stream.dict.get(b"DP").is_ok() {
            return Err(stream_error(
                page,
                "stream filter: /DecodeParms is not supported",
            ));
        }
        // The ASCII stages run through lopdf on a copy of the stream with the
        // trailing Flate filter removed, so what they hand on is at most the raw size.
        let packed = if head.is_empty() && flate {
            stream.content.clone()
        } else {
            let mut d = stream.dict.clone();
            if flate {
                d.set(
                    "Filter",
                    Object::Array(head.iter().map(|n| Object::Name(n.clone())).collect()),
                );
            }
            let mut s = lopdf::Stream::new(d, stream.content.clone());
            s.allows_compression = false;
            s.decompressed_content()
                .map_err(|e| stream_error(page, format!("stream filter: {e}")))?
        };
        if flate {
            use std::io::Read;
            let mut out = Vec::new();
            flate2::read::ZlibDecoder::new(packed.as_slice())
                .take(remaining.saturating_add(1))
                .read_to_end(&mut out)
                .map_err(|e| stream_error(page, format!("stream filter: {e}")))?;
            *used = used.saturating_add(out.len() as u64);
            if out.len() as u64 > remaining {
                return Err(over(*used));
            }
            out
        } else {
            *used = used.saturating_add(packed.len() as u64);
            if packed.len() as u64 > remaining {
                return Err(over(*used));
            }
            packed
        }
    };
    Ok(out)
}

/// The page's `/Contents` as one logical stream. A page's streams are separate
/// token runs, so they are joined with whitespace, never end to end.
fn page_content(
    doc: &Document,
    page_id: lopdf::ObjectId,
    page: usize,
    limits: &PdfVectorLimits,
    used: &mut u64,
) -> Result<Vec<u8>, PdfVectorError> {
    let mut out = Vec::new();
    let bad = || stream_error(page, "/Contents entry is not a stream");
    let Some(contents) = doc
        .get_dictionary(page_id)
        .ok()
        .and_then(|d| d.get(b"Contents").ok())
    else {
        return Ok(out);
    };
    fn one<'a>(
        doc: &'a Document,
        o: &'a Object,
        page: usize,
    ) -> Result<&'a lopdf::Stream, PdfVectorError> {
        let (_, obj) = doc
            .dereference(o)
            .map_err(|e| stream_error(page, format!("/Contents: {e}")))?;
        obj.as_stream()
            .map_err(|_| stream_error(page, "/Contents entry is not a stream"))
    }
    let (_, top) = doc
        .dereference(contents)
        .map_err(|e| stream_error(page, format!("/Contents: {e}")))?;
    let streams: Vec<&lopdf::Stream> = match top {
        Object::Array(items) => items
            .iter()
            .map(|o| one(doc, o, page))
            .collect::<Result<_, _>>()?,
        other => vec![other.as_stream().map_err(|_| bad())?],
    };
    for stream in streams {
        out.extend_from_slice(&stream_bytes(stream, page, limits, used)?);
        out.push(b'\n');
    }
    Ok(out)
}

/// `lopdf`'s decoder reads what it can and drops the rest, so an unterminated
/// string or array comes back as a shorter, valid-looking operation list. This
/// pass checks the lexical structure first, so a stream that does not tokenize
/// is an error and not a page with fewer paths.
///
/// The same pass counts operators and tracks `q` depth, so a stream that would trip
/// `max_operations` or `max_graphics_stack` is refused before `Content::decode` builds
/// an operation list for it, and what is held while refusing does not grow with the
/// stream. `ops_used` and `depth_base` are what the run has already spent on the page
/// and how deep the graphics stack already is where this stream starts.
fn check_tokens(
    bytes: &[u8],
    limits: &PdfVectorLimits,
    ops_used: u64,
    depth_base: usize,
) -> Result<(), Scan> {
    let (mut ops, mut depth) = (0u64, 0usize);
    fn is_space(b: u8) -> bool {
        matches!(b, 0 | b'\t' | b'\n' | 0x0c | b'\r' | b' ')
    }
    fn is_delim(b: u8) -> bool {
        matches!(
            b,
            b'(' | b')' | b'<' | b'>' | b'[' | b']' | b'{' | b'}' | b'/' | b'%'
        )
    }
    let (mut i, mut arrays, mut dicts) = (0usize, 0usize, 0usize);
    let mut in_inline_image = false;
    while i < bytes.len() {
        let b = bytes[i];
        match b {
            _ if is_space(b) => i += 1,
            b'%' => {
                while i < bytes.len() && bytes[i] != b'\n' && bytes[i] != b'\r' {
                    i += 1;
                }
            }
            b'(' => {
                let (mut depth, mut j) = (1usize, i + 1);
                while j < bytes.len() && depth > 0 {
                    match bytes[j] {
                        b'\\' => j += 1,
                        b'(' => depth += 1,
                        b')' => depth -= 1,
                        _ => {}
                    }
                    j += 1;
                }
                if depth > 0 {
                    return Err(Scan::Lex(format!("unterminated string at byte {i}")));
                }
                i = j;
            }
            b'<' if bytes.get(i + 1) == Some(&b'<') => {
                dicts += 1;
                i += 2;
            }
            b'<' => match bytes[i..].iter().position(|&c| c == b'>') {
                Some(n) => i += n + 1,
                None => return Err(Scan::Lex(format!("unterminated hex string at byte {i}"))),
            },
            b'>' if bytes.get(i + 1) == Some(&b'>') => {
                dicts = dicts
                    .checked_sub(1)
                    .ok_or_else(|| Scan::Lex(format!("`>>` without `<<` at byte {i}")))?;
                i += 2;
            }
            b'[' => {
                arrays += 1;
                i += 1;
            }
            b']' => {
                arrays = arrays
                    .checked_sub(1)
                    .ok_or_else(|| Scan::Lex(format!("`]` without `[` at byte {i}")))?;
                i += 1;
            }
            b')' | b'>' | b'{' | b'}' => i += 1,
            b'/' => {
                i += 1;
                while i < bytes.len() && !is_space(bytes[i]) && !is_delim(bytes[i]) {
                    i += 1;
                }
            }
            _ => {
                let start = i;
                while i < bytes.len() && !is_space(bytes[i]) && !is_delim(bytes[i]) {
                    i += 1;
                }
                if is_number_start(bytes[start]) {
                    check_number(&bytes[start..i]).map_err(Scan::Lex)?;
                }
                match &bytes[start..i] {
                    b"BI" => in_inline_image = true,
                    b"ID" if in_inline_image => {
                        // Binary data up to a whitespace-delimited `EI`.
                        let mut j = i;
                        let end = loop {
                            if j + 2 > bytes.len() {
                                return Err(Scan::Lex("inline image has no `EI`".to_string()));
                            }
                            if &bytes[j..j + 2] == b"EI"
                                && j > i
                                && is_space(bytes[j - 1])
                                && bytes.get(j + 2).is_none_or(|c| is_space(*c))
                            {
                                break j + 2;
                            }
                            j += 1;
                        };
                        in_inline_image = false;
                        i = end;
                    }
                    tok if !is_operand(tok) => {
                        ops += 1;
                        if ops_used + ops > limits.max_operations {
                            return Err(Scan::Operations(ops_used + ops));
                        }
                        match tok {
                            b"q" => {
                                depth += 1;
                                if depth_base + depth > limits.max_graphics_stack {
                                    return Err(Scan::Stack(depth_base + depth));
                                }
                            }
                            b"Q" => depth = depth.saturating_sub(1),
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    if arrays > 0 {
        return Err(Scan::Lex("unterminated array".to_string()));
    }
    if dicts > 0 {
        return Err(Scan::Lex("unterminated dictionary".to_string()));
    }
    Ok(())
}

/// Why the lexical pass refused a stream.
enum Scan {
    Lex(String),
    Operations(u64),
    Stack(usize),
}

fn is_number_start(b: u8) -> bool {
    b.is_ascii_digit() || matches!(b, b'+' | b'-' | b'.')
}

/// A PDF number is `[+-]digits[.digits]`. `lopdf` drops what it cannot parse and
/// carries on, so a malformed or out-of-range literal would otherwise read as a
/// shorter, valid-looking page.
fn check_number(tok: &[u8]) -> Result<(), String> {
    let text = String::from_utf8_lossy(tok);
    let body = tok
        .strip_prefix(b"+")
        .or(tok.strip_prefix(b"-"))
        .unwrap_or(tok);
    let (int, frac) = match body.iter().position(|&c| c == b'.') {
        Some(n) => (&body[..n], Some(&body[n + 1..])),
        None => (body, None),
    };
    let digits = |d: &[u8]| d.iter().all(u8::is_ascii_digit);
    if !digits(int)
        || !frac.is_none_or(digits)
        || (int.is_empty() && frac.is_none_or(<[u8]>::is_empty))
    {
        return Err(format!("malformed number `{text}`"));
    }
    if frac.is_none() && int.len() > 18 {
        return Err(format!("integer `{text}` is out of range"));
    }
    Ok(())
}

/// A bare token that is an operand and not an operator.
fn is_operand(tok: &[u8]) -> bool {
    matches!(tok, b"true" | b"false" | b"null" | b"BI" | b"ID" | b"EI")
        || tok
            .first()
            .is_some_and(|b| b.is_ascii_digit() || matches!(b, b'+' | b'-' | b'.'))
}

fn decode(
    bytes: &[u8],
    page: usize,
    limits: &PdfVectorLimits,
    ops_used: u64,
    depth_base: usize,
) -> Result<Vec<Operation>, PdfVectorError> {
    check_tokens(bytes, limits, ops_used, depth_base).map_err(|e| match e {
        Scan::Lex(e) => stream_error(page, format!("tokenize: {e}")),
        Scan::Operations(observed) => PdfVectorError::OperationLimitExceeded {
            limit: limits.max_operations,
            observed,
        },
        Scan::Stack(observed) => PdfVectorError::GraphicsStackExceeded {
            limit: limits.max_graphics_stack,
            observed,
        },
    })?;
    Content::decode(bytes)
        .map(|c| c.operations)
        .map_err(|e| stream_error(page, format!("tokenize: {e}")))
}

fn deref<'a>(doc: &'a Document, obj: &'a Object) -> Option<&'a Object> {
    match obj {
        Object::Reference(id) => doc.get_object(*id).ok(),
        other => Some(other),
    }
}

fn as_dict<'a>(doc: &'a Document, obj: &'a Object) -> Option<&'a Dictionary> {
    deref(doc, obj)?.as_dict().ok()
}

/// Follow a chain of indirect references to the dictionary it ends at. A chain that
/// meets an object twice, or runs past 64 links, is a cycle.
fn resolve_dict<'a>(
    doc: &'a Document,
    obj: &'a Object,
) -> Result<Option<&'a Dictionary>, PdfVectorError> {
    let mut seen: Vec<lopdf::ObjectId> = Vec::new();
    let mut cur = obj;
    loop {
        match cur {
            Object::Reference(id) => {
                if seen.contains(id) || seen.len() >= 64 {
                    return Err(PdfVectorError::ResourceCycle { object: id.0 });
                }
                seen.push(*id);
                // `objects.get`, not `get_object`: the latter chases references itself and
                // reports a loop as "not found", which would read as "no resources".
                match doc.objects.get(id) {
                    Some(next) => cur = next,
                    None => return Ok(None),
                }
            }
            Object::Dictionary(d) => return Ok(Some(d)),
            _ => return Ok(None),
        }
    }
}

/// `/Resources` of a page, inherited through `/Parent`. A `/Parent` chain that comes
/// back to a page-tree node it has already visited is a cycle, not "no resources".
fn inherited_resources(
    doc: &Document,
    page_id: lopdf::ObjectId,
) -> Result<Option<&Dictionary>, PdfVectorError> {
    let mut seen = vec![page_id];
    let mut current = page_id;
    loop {
        let Some(dict) = doc.get_object(current).ok().and_then(|o| o.as_dict().ok()) else {
            return Ok(None);
        };
        if let Ok(r) = dict.get(b"Resources") {
            return resolve_dict(doc, r);
        }
        current = match dict.get(b"Parent") {
            Ok(Object::Reference(id)) => *id,
            _ => return Ok(None),
        };
        if seen.contains(&current) {
            return Err(PdfVectorError::ResourceCycle { object: current.0 });
        }
        seen.push(current);
    }
}

fn number(o: &Object) -> Option<f64> {
    match o {
        Object::Integer(i) => Some(*i as f64),
        Object::Real(f) => Some(f64::from(*f)),
        _ => None,
    }
}

struct Run<'a> {
    doc: &'a Document,
    limits: &'a PdfVectorLimits,
    page: usize,
    /// Decoded content bytes spent so far, for `max_content_bytes`.
    content_bytes: u64,
    out: Vec<PdfPathEntity>,
    operations: u64,
    unbalanced: u32,
    shadings: u32,
    patterns: u32,
    form_path: Vec<String>,
    /// Saved graphics states. Shared across forms so the bound covers the whole expansion.
    stack: Vec<Saved>,
    /// Sequences of the clip paths in force, outermost first.
    clips: Vec<u64>,
    /// A non-`None` `/SMask` is in force.
    soft_mask: bool,
}

/// What `q` saves and `Q` restores.
struct Saved {
    ctm: Matrix,
    clips: usize,
    soft_mask: bool,
}

/// The path being built by one stream, in page space, plus the user-space current
/// point and subpath start that `v`, `y` and `h` need.
#[derive(Default)]
struct PathState {
    segments: Vec<PdfPathSegment>,
    current: Option<PdfPoint>,
    start: Option<PdfPoint>,
    clip: Option<PdfFillRule>,
}

impl Run<'_> {
    fn err(&self, detail: impl Into<String>) -> PdfVectorError {
        stream_error(self.page, detail)
    }

    fn operands(&self, op: &Operation, n: usize) -> Result<Vec<f64>, PdfVectorError> {
        if op.operands.len() != n {
            return Err(self.err(format!(
                "`{}` takes {n} operands, found {}",
                op.operator,
                op.operands.len()
            )));
        }
        op.operands
            .iter()
            .map(|o| {
                number(o)
                    .filter(|v| v.is_finite())
                    .ok_or_else(|| self.err(format!("`{}` has a non-numeric operand", op.operator)))
            })
            .collect()
    }

    fn stream(
        &mut self,
        ops: &[Operation],
        resources: Option<&Dictionary>,
        mut ctm: Matrix,
        depth: u32,
    ) -> Result<(), PdfVectorError> {
        let base = self.stack.len();
        let mut path = PathState::default();
        for op in ops {
            self.operations += 1;
            if self.operations > self.limits.max_operations {
                return Err(PdfVectorError::OperationLimitExceeded {
                    limit: self.limits.max_operations,
                    observed: self.operations,
                });
            }
            match op.operator.as_str() {
                "q" => {
                    let observed = self.stack.len() + 1;
                    if observed > self.limits.max_graphics_stack {
                        return Err(PdfVectorError::GraphicsStackExceeded {
                            limit: self.limits.max_graphics_stack,
                            observed,
                        });
                    }
                    self.stack.push(Saved {
                        ctm,
                        clips: self.clips.len(),
                        soft_mask: self.soft_mask,
                    });
                }
                "Q" => {
                    if self.stack.len() > base {
                        if let Some(saved) = self.stack.pop() {
                            ctm = saved.ctm;
                            self.clips.truncate(saved.clips);
                            self.soft_mask = saved.soft_mask;
                        }
                    } else {
                        self.unbalanced += 1;
                    }
                }
                "cm" => {
                    let v = self.operands(op, 6)?;
                    ctm = Matrix([v[0], v[1], v[2], v[3], v[4], v[5]]).then(&ctm);
                    if !ctm.0.iter().all(|c| c.is_finite()) {
                        return Err(self.err("`cm` overflows the current transformation matrix"));
                    }
                }
                "m" => {
                    let v = self.operands(op, 2)?;
                    let p = [v[0], v[1]];
                    path.segments.push(PdfPathSegment::MoveTo(ctm.apply(p)));
                    path.current = Some(p);
                    path.start = Some(p);
                }
                "l" => {
                    let v = self.operands(op, 2)?;
                    self.need_point(&path, "l")?;
                    let p = [v[0], v[1]];
                    path.segments.push(PdfPathSegment::LineTo(ctm.apply(p)));
                    path.current = Some(p);
                }
                "c" => {
                    let v = self.operands(op, 6)?;
                    self.need_point(&path, "c")?;
                    self.curve(&mut path, &ctm, [v[0], v[1]], [v[2], v[3]], [v[4], v[5]]);
                }
                "v" => {
                    // PDF 32000-1 8.5.2.2: the first control point is the current point.
                    let v = self.operands(op, 4)?;
                    let cur = self.need_point(&path, "v")?;
                    self.curve(&mut path, &ctm, cur, [v[0], v[1]], [v[2], v[3]]);
                }
                "y" => {
                    // The second control point is the end point.
                    let v = self.operands(op, 4)?;
                    self.need_point(&path, "y")?;
                    self.curve(&mut path, &ctm, [v[0], v[1]], [v[2], v[3]], [v[2], v[3]]);
                }
                "h" => {
                    self.operands(op, 0)?;
                    close(&mut path);
                }
                "re" => {
                    // PDF 32000-1 8.5.2.1: m, three l, h.
                    let v = self.operands(op, 4)?;
                    let (x, y, w, h) = (v[0], v[1], v[2], v[3]);
                    let corners = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]];
                    path.segments
                        .push(PdfPathSegment::MoveTo(ctm.apply(corners[0])));
                    for c in &corners[1..] {
                        path.segments.push(PdfPathSegment::LineTo(ctm.apply(*c)));
                    }
                    path.segments.push(PdfPathSegment::Close);
                    path.current = Some(corners[0]);
                    path.start = Some(corners[0]);
                }
                "W" => path.clip = Some(PdfFillRule::NonZero),
                "W*" => path.clip = Some(PdfFillRule::EvenOdd),
                "n" => self.paint(&mut path, op, None, false, false)?,
                "S" => self.paint(&mut path, op, None, true, false)?,
                "s" => self.paint(&mut path, op, None, true, true)?,
                "f" | "F" => self.paint(&mut path, op, Some(PdfFillRule::NonZero), false, false)?,
                "f*" => self.paint(&mut path, op, Some(PdfFillRule::EvenOdd), false, false)?,
                "B" => self.paint(&mut path, op, Some(PdfFillRule::NonZero), true, false)?,
                "B*" => self.paint(&mut path, op, Some(PdfFillRule::EvenOdd), true, false)?,
                "b" => self.paint(&mut path, op, Some(PdfFillRule::NonZero), true, true)?,
                "b*" => self.paint(&mut path, op, Some(PdfFillRule::EvenOdd), true, true)?,
                "gs" => self.ext_gstate(op, resources),
                "Do" => self.do_xobject(op, resources, &ctm, depth)?,
                "sh" => self.shadings = self.shadings.saturating_add(1),
                "scn" | "SCN" => {
                    if matches!(op.operands.last(), Some(Object::Name(_))) {
                        self.patterns = self.patterns.saturating_add(1);
                    }
                }
                // Colour, text, marked content and the rest are other tasks'.
                _ => {}
            }
            if path.segments.len() > self.limits.max_segments_per_path {
                return Err(PdfVectorError::PathSegmentsExceeded {
                    limit: self.limits.max_segments_per_path,
                    observed: path.segments.len(),
                });
            }
            if path.segments.iter().rev().take(5).any(|s| !finite(s)) {
                return Err(self.err(format!(
                    "`{}` produced a non-finite coordinate",
                    op.operator
                )));
            }
        }
        // A stream that leaves `q` open does not leak into its caller.
        self.stack.truncate(base);
        Ok(())
    }

    fn need_point(&self, path: &PathState, op: &str) -> Result<PdfPoint, PdfVectorError> {
        path.current
            .ok_or_else(|| self.err(format!("`{op}` with no current point")))
    }

    fn curve(&self, path: &mut PathState, ctm: &Matrix, c1: PdfPoint, c2: PdfPoint, end: PdfPoint) {
        path.segments.push(PdfPathSegment::CurveTo {
            c1: ctm.apply(c1),
            c2: ctm.apply(c2),
            end: ctm.apply(end),
        });
        path.current = Some(end);
    }

    fn paint(
        &mut self,
        path: &mut PathState,
        op: &Operation,
        fill: Option<PdfFillRule>,
        stroke: bool,
        close_first: bool,
    ) -> Result<(), PdfVectorError> {
        self.operands(op, 0)?;
        let mut state = std::mem::take(path);
        if close_first {
            close(&mut state);
        }
        if state.segments.is_empty() {
            return Ok(());
        }
        if self.out.len() >= self.limits.max_entities {
            return Err(PdfVectorError::EntityLimitExceeded {
                limit: self.limits.max_entities,
            });
        }
        let sequence = self.out.len() as u64;
        self.out.push(PdfPathEntity {
            segments: state.segments,
            paint: PdfPaintIntent {
                fill,
                stroke,
                close: close_first,
            },
            clip: state.clip,
            active_clips: self.clips.clone(),
            soft_masked: self.soft_mask,
            provenance: PdfProvenance {
                page: self.page,
                operator: op.operator.clone(),
                form_path: self.form_path.clone(),
                sequence,
            },
        });
        if state.clip.is_some() {
            self.clips.push(sequence);
        }
        Ok(())
    }

    /// `gs`: only `/SMask` is read here. A name or dictionary that is missing leaves the
    /// state as it was, as for any other entry this adapter does not interpret.
    fn ext_gstate(&mut self, op: &Operation, resources: Option<&Dictionary>) {
        let [Object::Name(name)] = op.operands.as_slice() else {
            return;
        };
        let doc = self.doc;
        let Some(gs) = resources
            .and_then(|r| r.get(b"ExtGState").ok())
            .and_then(|x| as_dict(doc, x))
            .and_then(|d| d.get(name).ok())
            .and_then(|o| as_dict(doc, o))
        else {
            return;
        };
        match gs.get(b"SMask").ok().and_then(|m| deref(doc, m)) {
            Some(Object::Name(n)) if n == b"None" => self.soft_mask = false,
            Some(Object::Dictionary(_) | Object::Stream(_)) => self.soft_mask = true,
            _ => {}
        }
    }

    fn do_xobject(
        &mut self,
        op: &Operation,
        resources: Option<&Dictionary>,
        ctm: &Matrix,
        depth: u32,
    ) -> Result<(), PdfVectorError> {
        let [Object::Name(name)] = op.operands.as_slice() else {
            return Err(self.err("`Do` takes one name operand"));
        };
        let doc = self.doc;
        let Some(xobj) = resources
            .and_then(|r| r.get(b"XObject").ok())
            .and_then(|x| as_dict(doc, x))
            .and_then(|d| d.get(name).ok())
            .and_then(|o| deref(doc, o))
            .and_then(|o| o.as_stream().ok())
        else {
            // Not found, or not a stream: nothing to expand, as a viewer would do.
            return Ok(());
        };
        let is_form = xobj
            .dict
            .get(b"Subtype")
            .ok()
            .and_then(|s| s.as_name().ok())
            .is_some_and(|n| n == b"Form");
        if !is_form {
            return Ok(());
        }
        let observed = depth + 1;
        if observed > self.limits.max_form_depth {
            return Err(PdfVectorError::FormDepthExceeded {
                limit: self.limits.max_form_depth,
                observed,
            });
        }
        let matrix = match xobj.dict.get(b"Matrix").ok().and_then(|m| deref(doc, m)) {
            None => Matrix::IDENTITY,
            Some(Object::Array(a)) if a.len() == 6 => {
                let mut m = [0.0; 6];
                for (slot, o) in m.iter_mut().zip(a) {
                    *slot = deref(doc, o)
                        .and_then(number)
                        .filter(|v| v.is_finite())
                        .ok_or_else(|| self.err("form /Matrix has a non-numeric entry"))?;
                }
                Matrix(m)
            }
            Some(_) => return Err(self.err("form /Matrix is not six numbers")),
        };
        let bytes = stream_bytes(xobj, self.page, self.limits, &mut self.content_bytes)?;
        let ops = decode(
            &bytes,
            self.page,
            self.limits,
            self.operations,
            self.stack.len(),
        )?;
        drop(bytes);
        // The form's own resources, else the invoker's (PDF 32000-1 7.8.3).
        let own = match xobj.dict.get(b"Resources") {
            Ok(r) => resolve_dict(doc, r)?,
            Err(_) => None,
        };
        self.form_path
            .push(String::from_utf8_lossy(name).into_owned());
        // A form runs as if inside `q`/`Q`: what it clips or masks ends with it.
        let (clips, soft_mask) = (self.clips.len(), self.soft_mask);
        let r = self.stream(&ops, own.or(resources), matrix.then(ctm), observed);
        self.clips.truncate(clips);
        self.soft_mask = soft_mask;
        self.form_path.pop();
        r
    }
}

fn finite(seg: &PdfPathSegment) -> bool {
    match seg {
        PdfPathSegment::MoveTo(p) | PdfPathSegment::LineTo(p) => p.iter().all(|c| c.is_finite()),
        PdfPathSegment::CurveTo { c1, c2, end } => [c1, c2, end]
            .iter()
            .all(|p| p.iter().all(|c| c.is_finite())),
        _ => true,
    }
}

fn close(path: &mut PathState) {
    if path.segments.is_empty() {
        return;
    }
    path.segments.push(PdfPathSegment::Close);
    path.current = path.start;
}
