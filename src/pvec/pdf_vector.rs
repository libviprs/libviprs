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

use super::text_policy::{TextFontOutcome, TextReport, font_outcome};
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
    /// Text-showing operators seen and omitted, by operator and font outcome (N5.5).
    pub text: TextReport,
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
    let base = Matrix::translate(-l, -b).then(&rotation_matrix(rotation, w, h));
    let (width, height) = match rotation {
        PageRotation::Quarter | PageRotation::ThreeQuarter => (h, w),
        _ => (w, h),
    };

    let content = page_content(&doc, page_id, page)?;
    let ops = decode(&content, page)?;
    let mut run = Run {
        doc: &doc,
        limits,
        page,
        out: Vec::new(),
        operations: 0,
        unbalanced: 0,
        form_path: Vec::new(),
        stack: Vec::new(),
        clips: Vec::new(),
        soft_mask: false,
        text: TextReport::default(),
        font: TextFontOutcome::Unresolved,
    };
    let resources = inherited_resources(&doc, page_id);
    run.stream(&ops, resources, base, 0)?;

    Ok(PdfVectorPage {
        page,
        width,
        height,
        rotation_degrees: rotation.as_degrees(),
        entities: run.out,
        unbalanced_restores: run.unbalanced,
        text: run.text,
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

/// The text of one content stream: decoded when it has a filter, as stored when it
/// has none. A filter that fails is an error, not a fall back to the raw bytes.
fn stream_bytes(stream: &lopdf::Stream, page: usize) -> Result<Vec<u8>, PdfVectorError> {
    if stream.dict.has(b"Filter") {
        stream
            .decompressed_content()
            .map_err(|e| stream_error(page, format!("stream filter: {e}")))
    } else {
        Ok(stream.content.clone())
    }
}

/// The page's `/Contents` as one logical stream. A page's streams are separate
/// token runs, so they are joined with whitespace, never end to end.
fn page_content(
    doc: &Document,
    page_id: lopdf::ObjectId,
    page: usize,
) -> Result<Vec<u8>, PdfVectorError> {
    let mut out = Vec::new();
    for id in doc.get_page_contents(page_id) {
        let obj = doc
            .get_object(id)
            .map_err(|e| stream_error(page, format!("/Contents: {e}")))?;
        let stream = obj
            .as_stream()
            .map_err(|_| stream_error(page, "/Contents entry is not a stream"))?;
        out.extend_from_slice(&stream_bytes(stream, page)?);
        out.push(b'\n');
    }
    Ok(out)
}

/// `lopdf`'s decoder reads what it can and drops the rest, so an unterminated
/// string or array comes back as a shorter, valid-looking operation list. This
/// pass checks the lexical structure first, so a stream that does not tokenize
/// is an error and not a page with fewer paths.
fn check_tokens(bytes: &[u8]) -> Result<(), String> {
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
                    return Err(format!("unterminated string at byte {i}"));
                }
                i = j;
            }
            b'<' if bytes.get(i + 1) == Some(&b'<') => {
                dicts += 1;
                i += 2;
            }
            b'<' => match bytes[i..].iter().position(|&c| c == b'>') {
                Some(n) => i += n + 1,
                None => return Err(format!("unterminated hex string at byte {i}")),
            },
            b'>' if bytes.get(i + 1) == Some(&b'>') => {
                dicts = dicts
                    .checked_sub(1)
                    .ok_or_else(|| format!("`>>` without `<<` at byte {i}"))?;
                i += 2;
            }
            b'[' => {
                arrays += 1;
                i += 1;
            }
            b']' => {
                arrays = arrays
                    .checked_sub(1)
                    .ok_or_else(|| format!("`]` without `[` at byte {i}"))?;
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
                match &bytes[start..i] {
                    b"BI" => in_inline_image = true,
                    b"ID" if in_inline_image => {
                        // Binary data up to a whitespace-delimited `EI`.
                        let mut j = i;
                        let end = loop {
                            if j + 2 > bytes.len() {
                                return Err("inline image has no `EI`".to_string());
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
                    _ => {}
                }
            }
        }
    }
    if arrays > 0 {
        return Err("unterminated array".to_string());
    }
    if dicts > 0 {
        return Err("unterminated dictionary".to_string());
    }
    Ok(())
}

fn decode(bytes: &[u8], page: usize) -> Result<Vec<Operation>, PdfVectorError> {
    check_tokens(bytes).map_err(|e| stream_error(page, format!("tokenize: {e}")))?;
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

/// `/Resources` of a page, inherited through `/Parent`.
fn inherited_resources(doc: &Document, page_id: lopdf::ObjectId) -> Option<&Dictionary> {
    let mut current = page_id;
    for _ in 0..64 {
        let dict = doc.get_object(current).ok()?.as_dict().ok()?;
        if let Ok(r) = dict.get(b"Resources") {
            return as_dict(doc, r);
        }
        current = match dict.get(b"Parent") {
            Ok(Object::Reference(id)) if *id != current => *id,
            _ => return None,
        };
    }
    None
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
    out: Vec<PdfPathEntity>,
    operations: u64,
    unbalanced: u32,
    form_path: Vec<String>,
    /// Saved graphics states. Shared across forms so the bound covers the whole expansion.
    stack: Vec<Saved>,
    /// Sequences of the clip paths in force, outermost first.
    clips: Vec<u64>,
    /// A non-`None` `/SMask` is in force.
    soft_mask: bool,
    /// Text omitted so far (N5.5).
    text: TextReport,
    /// The font the last `Tf` named, part of the graphics state.
    font: TextFontOutcome,
}

/// What `q` saves and `Q` restores.
struct Saved {
    ctm: Matrix,
    clips: usize,
    soft_mask: bool,
    font: TextFontOutcome,
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
                        font: self.font,
                    });
                }
                "Q" => {
                    if self.stack.len() > base {
                        if let Some(saved) = self.stack.pop() {
                            ctm = saved.ctm;
                            self.clips.truncate(saved.clips);
                            self.soft_mask = saved.soft_mask;
                            self.font = saved.font;
                        }
                    } else {
                        self.unbalanced += 1;
                    }
                }
                "cm" => {
                    let v = self.operands(op, 6)?;
                    ctm = Matrix([v[0], v[1], v[2], v[3], v[4], v[5]]).then(&ctm);
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
                "Tf" => self.font = font_outcome(self.doc, resources, op.operands.first()),
                // Text is omitted and counted (N5.5, docs/pvec-text-policy.md).
                "Tj" | "TJ" | "'" | "\"" => self.text.record_pdf(&op.operator, self.font),
                // Colour, shading, marked content and the rest are other tasks'.
                _ => {}
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
        let bytes = stream_bytes(xobj, self.page)?;
        let ops = decode(&bytes, self.page)?;
        // The form's own resources, else the invoker's (PDF 32000-1 7.8.3).
        let own = xobj
            .dict
            .get(b"Resources")
            .ok()
            .and_then(|r| as_dict(doc, r));
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

fn close(path: &mut PathState) {
    if path.segments.is_empty() {
        return;
    }
    path.segments.push(PdfPathSegment::Close);
    path.current = path.start;
}
