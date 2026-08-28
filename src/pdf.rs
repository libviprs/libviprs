use std::path::Path;

use thiserror::Error;

use crate::pixel::PixelFormat;
use crate::raster::Raster;
use crate::source;

/// Errors that can occur during PDF inspection, image extraction, or rendering.
///
/// Covers I/O failures, PDF parsing errors, missing or unsupported images,
/// page-range validation, and (when the `pdfium` feature is enabled) pdfium
/// rendering failures.
///
/// # Examples
///
/// See [pdf_ops tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pdf_ops.rs)
/// for error handling patterns.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-render)
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PdfError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("PDF parse error: {0}")]
    Parse(String),
    #[error("no image found on page {page}")]
    NoImageOnPage { page: usize },
    #[error("unsupported image format in PDF: {0}")]
    UnsupportedFormat(String),
    #[error("image decode error: {0}")]
    Decode(String),
    #[error("decompressed stream exceeded {limit}-byte cap (possible zip bomb)")]
    DecompressionLimitExceeded { limit: usize },
    #[error("raster error: {0}")]
    Raster(#[from] crate::raster::RasterError),
    #[error("page {page} out of range (document has {total} pages)")]
    PageOutOfRange { page: usize, total: usize },
    #[error(
        "document has {count} pages, exceeding pdfium's {max}-page index limit (c_int page count)"
    )]
    PageCountExceedsIndex { count: usize, max: usize },
    #[error("pdfium error: {0}")]
    Pdfium(String),
    #[error(
        "budget exceeded: worst-case strip {strip_bytes} bytes > budget {budget_bytes} bytes (at DPI {dpi})"
    )]
    BudgetExceeded {
        strip_bytes: u64,
        budget_bytes: u64,
        dpi: u32,
    },
    #[error("unsupported page /Rotate value: {0} (must be a multiple of 90)")]
    UnsupportedRotation(i64),
    #[error("render exceeds pixel budget: {pixels} px (width × height) > {budget} px ceiling")]
    RenderBudgetExceeded { pixels: u64, budget: u64 },
    #[error("failed to allocate {bytes} bytes for render buffer")]
    AllocationFailed { bytes: usize },
    #[error(
        "render dimensions {width}x{height} exceed pdfium's i32 bitmap-span limit ({span} bytes > i32::MAX)"
    )]
    RenderTooLarge { width: u32, height: u32, span: u64 },
}

/// Default ceiling on the pixel count (`width × height`) that a single
/// non-budgeted pdfium render is permitted to allocate.
///
/// [`render_page_pdfium`] and the streaming source constructor derive the
/// output bitmap size from the page's `/MediaBox` times the requested DPI
/// via a saturating `f64 as u32` cast. A crafted `/MediaBox` at a normal or
/// high DPI can drive that product into the billions of pixels, so the raw
/// pdfium bitmap plus the RGBA [`Raster`] copy would attempt a multi-gigabyte
/// allocation and OOM-abort the process before any downstream size check
/// runs. Bounding the pixel count up front converts that abort into a
/// recoverable [`PdfError::RenderBudgetExceeded`].
///
/// The ceiling is a backstop against clearly-adversarial sizes (a crafted
/// `/MediaBox` at a high DPI reaches hundreds of billions of pixels), not a
/// tight bound: legitimate large-format renders — e.g. a 48"×36" AutoCAD
/// blueprint at 300 DPI is ~518 megapixels — must still succeed. `2^30` px is
/// 4 GiB for the RGBA raster alone; callers that need a tighter, DPI-reducing
/// bound use [`render_page_pdfium_budgeted`] with an explicit `max_pixels`.
pub const DEFAULT_MAX_RENDER_PIXELS: u64 = 1 << 30;

/// Convert page dimensions in points into a render size in pixels at `dpi`,
/// rejecting any size whose total pixel count exceeds `max_pixels`.
///
/// The two dimension casts saturate (`f32 as u32`) and the product is taken
/// in `u64`, so an adversarial `/MediaBox` can never wrap into a small value.
/// Returns [`PdfError::RenderBudgetExceeded`] when the budget is exceeded so
/// the caller propagates a typed error instead of proceeding to a
/// multi-gigabyte allocation.
#[cfg(any(feature = "pdfium", test))]
pub(crate) fn render_dims_within_budget(
    width_pt: f32,
    height_pt: f32,
    dpi: u32,
    max_pixels: u64,
) -> Result<(u32, u32), PdfError> {
    let scale = dpi as f32 / 72.0;
    let width = (width_pt * scale) as u32;
    let height = (height_pt * scale) as u32;
    let pixels = width as u64 * height as u64;
    if pixels > max_pixels {
        return Err(PdfError::RenderBudgetExceeded {
            pixels,
            budget: max_pixels,
        });
    }
    Ok((width, height))
}

/// Choose the render DPI so the rasterized page fits within `max_pixels`,
/// preferring `max_dpi` when the page already fits. Returns
/// `(dpi_used, capped)`, where `capped` is true when the DPI was reduced to
/// honor the budget.
///
/// The pixel estimate is taken in `f64`: an adversarial `/MediaBox` large
/// enough to overflow a `u64` product would otherwise wrap to a small value
/// that slips under `max_pixels`, skipping the reduction branch and running
/// the render at full DPI — the exact OOM the budget exists to prevent. In
/// `f64` an overflowing product saturates toward `inf` and compares correctly.
#[cfg(any(feature = "pdfium", test))]
pub(crate) fn budgeted_render_dpi(
    width_pts: f64,
    height_pts: f64,
    max_dpi: u32,
    max_pixels: u64,
) -> (u32, bool) {
    let scale_at_max = max_dpi as f64 / 72.0;
    let pixels_at_max = (width_pts * scale_at_max) * (height_pts * scale_at_max);

    if pixels_at_max <= max_pixels as f64 {
        (max_dpi, false)
    } else {
        // scale = sqrt(max_pixels / (w_pts * h_pts)), then dpi = scale * 72
        let scale = (max_pixels as f64 / (width_pts * height_pts)).sqrt();
        let dpi = (scale * 72.0).floor() as u32;
        (dpi.max(1), true)
    }
}

/// Allocate a zeroed RGBA8 buffer of `width × height` pixels using fallible
/// allocation, so an adversarial size yields a typed error rather than an
/// allocator abort.
///
/// The byte length is computed with checked arithmetic to reject a `usize`
/// overflow up front (`width × height × 4` can exceed `usize::MAX` for
/// adversarial `u32` dimensions on 64-bit targets), and [`Vec::try_reserve`]
/// turns an allocation failure into [`PdfError::AllocationFailed`] instead of
/// aborting.
#[cfg(any(feature = "pdfium", test))]
pub(crate) fn alloc_zeroed_rgba(width: u32, height: u32) -> Result<Vec<u8>, PdfError> {
    let len = (width as usize)
        .checked_mul(height as usize)
        .and_then(|px| px.checked_mul(4))
        .ok_or(PdfError::RenderBudgetExceeded {
            pixels: u64::from(width).saturating_mul(u64::from(height)),
            budget: DEFAULT_MAX_RENDER_PIXELS,
        })?;
    let mut buf: Vec<u8> = Vec::new();
    buf.try_reserve(len)
        .map_err(|_| PdfError::AllocationFailed { bytes: len })?;
    buf.resize(len, 0);
    Ok(buf)
}

/// A page's intrinsic `/Rotate` value, normalised to one of the four
/// values the PDF spec admits.
///
/// PDF 1.7 §7.7.3.3 defines `/Rotate` as a multiple of 90 degrees;
/// any value outside `{0, 90, 180, 270}` after normalisation
/// (`rem_euclid 360`) is malformed. This enum makes the well-formed
/// values type-level and lets the matrix-render code path drop the
/// otherwise-dead "unsupported rotation" branch.
///
/// Construct via [`Self::try_from_degrees`] for parsing, or directly
/// match on the four variants when handling all rotations.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PageRotation {
    /// No rotation. The page renders in its authored orientation.
    #[default]
    Zero,
    /// 90° clockwise. Portrait page renders as landscape; the page's
    /// top edge becomes the displayed right edge.
    Quarter,
    /// 180°. Same orientation as `Zero` but flipped about both axes.
    Half,
    /// 270° clockwise (equivalently 90° counter-clockwise). Portrait
    /// renders as landscape, with the page's top edge becoming the
    /// displayed left edge.
    ThreeQuarter,
}

impl PageRotation {
    /// Map a degree value (typically from `/Rotate`) to a [`PageRotation`].
    /// The input is normalised via `rem_euclid 360` so negatives and
    /// values ≥360 are accepted as long as they're a multiple of 90.
    ///
    /// # Errors
    ///
    /// [`PdfError::UnsupportedRotation`] for any value whose normalised
    /// form is not in `{0, 90, 180, 270}`.
    pub fn try_from_degrees(degrees: i64) -> Result<Self, PdfError> {
        match degrees.rem_euclid(360) {
            0 => Ok(Self::Zero),
            90 => Ok(Self::Quarter),
            180 => Ok(Self::Half),
            270 => Ok(Self::ThreeQuarter),
            _ => Err(PdfError::UnsupportedRotation(degrees)),
        }
    }

    /// Inverse of [`Self::try_from_degrees`]: returns 0, 90, 180, or 270.
    #[inline]
    #[must_use]
    pub const fn as_degrees(self) -> i64 {
        match self {
            Self::Zero => 0,
            Self::Quarter => 90,
            Self::Half => 180,
            Self::ThreeQuarter => 270,
        }
    }
}

/// Information about a PDF document, including page count and per-page metadata.
///
/// Returned by [`pdf_info`]. Use this to inspect a PDF before deciding whether
/// to extract embedded images or render pages with pdfium.
///
/// # Examples
///
/// See [pdf_ops tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pdf_ops.rs)
/// and the [CLI info command](https://github.com/libviprs/libviprs-cli/blob/main/src/main.rs).
///
/// **See also:** [interactive example](https://libviprs.org/cli/#info)
#[derive(Debug, Clone)]
pub struct PdfInfo {
    pub page_count: usize,
    pub pages: Vec<PdfPageInfo>,
}

/// Metadata for a single page within a PDF document.
///
/// Dimensions are in PDF points (1 point = 1/72 inch). To convert to pixels
/// at a given DPI, multiply by `dpi / 72.0`. The `has_images` flag indicates
/// whether the page contains embedded raster images that can be extracted
/// with [`extract_page_image`].
///
/// # Examples
///
/// See [pdf_ops tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pdf_ops.rs).
///
/// **See also:** [interactive example](https://libviprs.org/cli/#info)
#[derive(Debug, Clone)]
pub struct PdfPageInfo {
    pub page_number: usize,
    /// Page width in PDF points (1/72 inch).
    pub width_pts: f64,
    /// Page height in PDF points (1/72 inch).
    pub height_pts: f64,
    /// Whether the page contains embedded raster images.
    pub has_images: bool,
}

/// Get information about a PDF document, including page count and per-page
/// dimensions and image presence.
///
/// Use this to inspect a PDF before extracting images or rendering pages.
/// For scanned blueprints, check [`PdfPageInfo::has_images`] to decide
/// whether to use [`extract_page_image`] (fast, embedded image extraction)
/// or [`render_page_pdfium`] (full vector rendering).
///
/// # Examples
///
/// See [pdf_ops tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pdf_ops.rs)
/// and the [CLI info command](https://github.com/libviprs/libviprs-cli/blob/main/src/main.rs).
///
/// **See also:** [interactive example](https://libviprs.org/cli/#info)
pub fn pdf_info(path: &Path) -> Result<PdfInfo, PdfError> {
    let doc = lopdf::Document::load(path).map_err(|e| PdfError::Parse(e.to_string()))?;
    Ok(pdf_info_from_doc(&doc))
}

/// Build a [`PdfInfo`] from an already-loaded document. Shared by [`pdf_info`]
/// and [`pdf_info_with_password`] so the loaded document is walked once.
fn pdf_info_from_doc(doc: &lopdf::Document) -> PdfInfo {
    let pages_map = doc.get_pages();
    let page_count = pages_map.len();
    let mut pages = Vec::with_capacity(page_count);

    // Pages are returned as BTreeMap<u32, ObjectId>, sorted by page number
    for (&page_num, &page_id) in &pages_map {
        let (width_pts, height_pts) = get_page_dimensions(doc, page_id);
        let has_images = page_has_images(doc, page_id);

        pages.push(PdfPageInfo {
            page_number: page_num as usize,
            width_pts,
            height_pts,
            has_images,
        });
    }

    PdfInfo { page_count, pages }
}

/// Build a shared decode error naming a PDF capability that is not available
/// in this build.
fn decode_unavailable(what: impl std::fmt::Display) -> crate::codec::DecodeError {
    crate::source::SourceError::Io(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        format!("{what} is not available in this build"),
    ))
}

/// Fold a [`PdfError`] into the shared decode error, so the DPI/password
/// extract helpers can report through the `DecodeError` their contracts name.
fn pdf_to_decode(err: PdfError) -> crate::codec::DecodeError {
    crate::source::SourceError::Io(std::io::Error::other(err.to_string()))
}

/// Read [`PdfInfo`] from a password-protected PDF (libvips `pdfload` with
/// `password`).
///
/// Unencrypted documents open regardless of `password`, so this returns the
/// same result as [`pdf_info`] for them. Encrypted documents need a decryption
/// path this pure-Rust build does not provide: only when the opened document
/// actually carries an encryption dictionary (`lopdf`'s
/// [`is_encrypted`](lopdf::Document::is_encrypted)) and a non-empty `password`
/// was supplied does this report a typed [`PdfError::UnsupportedFormat`] naming
/// the password-protected case. A missing, unreadable, or malformed file
/// surfaces its real IO/parse error unchanged rather than being mislabelled as
/// password-protected.
///
/// # Errors
///
/// [`PdfError::UnsupportedFormat`] for an encrypted document opened with a
/// non-empty password, otherwise the same errors as [`pdf_info`].
pub fn pdf_info_with_password(path: &Path, password: &str) -> Result<PdfInfo, PdfError> {
    // Open first so a missing/unreadable/malformed file surfaces its real
    // IO/parse error. `lopdf` loads an encrypted document's structure without
    // decrypting its streams, so the encryption dictionary is observable here.
    let doc = lopdf::Document::load(path).map_err(|e| PdfError::Parse(e.to_string()))?;
    if !password.is_empty() && doc.is_encrypted() {
        return Err(PdfError::UnsupportedFormat(
            "password-protected PDF decryption is not available in this build".to_string(),
        ));
    }
    Ok(pdf_info_from_doc(&doc))
}

/// Extract a page's embedded image at a target render DPI (libvips `pdfload`
/// with `dpi`), page numbers 1-based.
///
/// With the `pdfium` feature the page is rendered at `dpi` through pdfium, so
/// the output dimensions scale with the DPI. Without `pdfium` there is no
/// DPI-controlled rasteriser, so this reports a typed decode error: embedded
/// extraction ([`extract_page_image`]) returns images at their stored size
/// and cannot honour a DPI.
///
/// # Errors
///
/// A [`crate::codec::DecodeError`]: an unsupported-capability error without
/// `pdfium`, an invalid-input error for a non-positive DPI or a zero page,
/// otherwise the rasteriser's error folded into the decode error.
pub fn extract_page_image_dpi(
    path: &Path,
    page: u32,
    dpi: f64,
) -> Result<Raster, crate::codec::DecodeError> {
    #[cfg(feature = "pdfium")]
    {
        if page == 0 {
            return Err(crate::source::SourceError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "page index must be >= 1",
            )));
        }
        if !(dpi.is_finite() && dpi >= 1.0) {
            return Err(crate::source::SourceError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("dpi must be finite and >= 1, got {dpi}"),
            )));
        }
        let dpi_u32 = dpi.round() as u32;
        render_page_pdfium(path, page as usize, dpi_u32).map_err(pdf_to_decode)
    }
    #[cfg(not(feature = "pdfium"))]
    {
        let _ = (path, page, dpi);
        Err(decode_unavailable(
            "PDF rendering at a specified DPI (requires the `pdfium` feature)",
        ))
    }
}

/// A typed background fill colour for [`extract_page_image_with_background_typed`].
///
/// [`extract_page_image_with_background`] takes a loosely-typed `&[f64]` whose
/// "3 (`r, g, b`) or 4 (`r, g, b, a`) channels" contract is only checked at
/// runtime. This enum lifts that contract into the type system: a background is
/// either three opaque RGB channels or four RGBA channels, each a `0..=255`
/// `u8` intensity, so a wrong channel count is unrepresentable rather than an
/// invalid-input error.
///
/// A `[u8; 3]` or `[u8; 4]` array converts in via [`From`]
/// ([`Rgb`](Self::Rgb) is implicitly opaque), so callers can hand an array
/// literal straight to [`extract_page_image_with_background_typed`], which
/// accepts any `impl Into<BackgroundColor>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackgroundColor {
    /// Opaque `[r, g, b]`; the alpha channel is implicitly `255` (fully
    /// opaque), matching a 3-channel `&[f64]` background.
    Rgb([u8; 3]),
    /// `[r, g, b, a]` with an explicit `0..=255` alpha channel.
    Rgba([u8; 4]),
}

impl From<[u8; 3]> for BackgroundColor {
    fn from(rgb: [u8; 3]) -> Self {
        BackgroundColor::Rgb(rgb)
    }
}

impl From<[u8; 4]> for BackgroundColor {
    fn from(rgba: [u8; 4]) -> Self {
        BackgroundColor::Rgba(rgba)
    }
}

/// DPI at which [`extract_page_image_with_background`] rasterises a page.
///
/// The background helper takes no DPI parameter, so it renders at libvips
/// `pdfload`'s 72-DPI baseline (the resolution at which one PDF point maps to
/// one device pixel). Callers that need a specific resolution reach for
/// [`extract_page_image_dpi`], which this helper otherwise mirrors.
#[cfg(feature = "pdfium")]
const DEFAULT_BACKGROUND_RENDER_DPI: u32 = 72;

/// Extract a page image over a solid background fill (libvips `pdfload` with a
/// `background`), page numbers 1-based.
///
/// pdfium clears the output bitmap to `background` before drawing the page, so
/// any region the page leaves transparent shows `background` instead of the
/// default white. `background` is read as `[r, g, b]` or `[r, g, b, a]` with
/// each channel a `0..=255` intensity; a missing alpha defaults to fully
/// opaque. Out-of-range values are clamped into `0..=255` and a non-finite
/// channel falls back to its default, so a NaN alpha stays opaque and the u8
/// narrowing never wraps. The page renders at libvips `pdfload`'s 72-DPI
/// baseline; callers that need a specific resolution use
/// [`extract_page_image_dpi`], which this mirrors.
///
/// # Errors
///
/// A [`crate::codec::DecodeError`]: an unsupported-capability error without
/// `pdfium`, an invalid-input error for a zero page or a `background` that is
/// not 3 (`r, g, b`) or 4 (`r, g, b, a`) channels, otherwise the rasteriser's
/// error folded into the decode error.
pub fn extract_page_image_with_background(
    path: &Path,
    page: u32,
    background: &[f64],
) -> Result<Raster, crate::codec::DecodeError> {
    #[cfg(feature = "pdfium")]
    {
        if page == 0 {
            return Err(crate::source::SourceError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "page index must be >= 1",
            )));
        }
        if !(3..=4).contains(&background.len()) {
            return Err(crate::source::SourceError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "background must supply 3 (r, g, b) or 4 (r, g, b, a) channels, got {}",
                    background.len()
                ),
            )));
        }
        render_page_pdfium_with_background(
            path,
            page as usize,
            DEFAULT_BACKGROUND_RENDER_DPI,
            background,
        )
        .map_err(pdf_to_decode)
    }
    #[cfg(not(feature = "pdfium"))]
    {
        let _ = (path, page, background);
        Err(decode_unavailable(
            "PDF rendering with a background fill (requires the `pdfium` feature)",
        ))
    }
}

/// Extract a page image over a solid background fill, the typed counterpart of
/// [`extract_page_image_with_background`], page numbers 1-based.
///
/// Identical to [`extract_page_image_with_background`] except the background is
/// a [`BackgroundColor`] whose channel count is fixed by its type rather than a
/// `&[f64]` slice validated at runtime. `background` accepts any
/// `impl Into<BackgroundColor>`, so a `[u8; 3]` (opaque RGB) or `[u8; 4]`
/// (RGBA) array literal passes straight in.
///
/// The colour is forwarded verbatim to [`extract_page_image_with_background`]
/// as the equivalent 3- or 4-channel `&[f64]` slice, so the rendered result is
/// byte-identical to the loosely-typed entry point for the same colour. The
/// page renders at the same 72-DPI baseline; see
/// [`extract_page_image_with_background`] for the clear-colour and DPI details.
///
/// # Errors
///
/// The same [`crate::codec::DecodeError`] as
/// [`extract_page_image_with_background`]: an unsupported-capability error
/// without `pdfium`, an invalid-input error for a zero page, otherwise the
/// rasteriser's error folded into the decode error. The channel-count
/// validation the slice form performs cannot fire here — [`BackgroundColor`]
/// makes an invalid channel count unrepresentable.
pub fn extract_page_image_with_background_typed(
    path: &Path,
    page: u32,
    background: impl Into<BackgroundColor>,
) -> Result<Raster, crate::codec::DecodeError> {
    // Delegate to the `&[f64]` entry point with the equivalent 3- or 4-channel
    // slice so the result is byte-identical to the loosely-typed form, and the
    // non-`pdfium` build inherits its typed unsupported-capability error.
    match background.into() {
        BackgroundColor::Rgb(rgb) => {
            extract_page_image_with_background(path, page, &rgb.map(f64::from))
        }
        BackgroundColor::Rgba(rgba) => {
            extract_page_image_with_background(path, page, &rgba.map(f64::from))
        }
    }
}

/// Extract a page image from a password-protected PDF (libvips `pdfload` with
/// `password`), page numbers 1-based.
///
/// Unencrypted documents extract regardless of `password`. Encrypted documents
/// need a decryption path this build does not provide: only when the opened
/// document actually carries an encryption dictionary (`lopdf`'s
/// [`is_encrypted`](lopdf::Document::is_encrypted)) and a non-empty `password`
/// was supplied does this report a typed unsupported-capability decode error
/// naming the password-protected case. A missing, unreadable, or malformed file
/// surfaces its real IO/parse error folded into the decode error rather than
/// being mislabelled as password-protected.
///
/// # Errors
///
/// A [`crate::codec::DecodeError`]: an unsupported-capability error for an
/// encrypted document opened with a non-empty password, otherwise the
/// extraction error folded into the decode error.
pub fn extract_page_image_with_password(
    path: &Path,
    page: u32,
    password: &str,
) -> Result<Raster, crate::codec::DecodeError> {
    // Open first so a missing/unreadable/malformed file surfaces its real
    // error. Only a genuinely encrypted document folds to the
    // password-protected capability error.
    let doc =
        lopdf::Document::load(path).map_err(|e| pdf_to_decode(PdfError::Parse(e.to_string())))?;
    if !password.is_empty() && doc.is_encrypted() {
        return Err(decode_unavailable("password-protected PDF decryption"));
    }
    extract_page_image_from_doc(&doc, page as usize).map_err(pdf_to_decode)
}

/// Resolve a 1-based `page` number to its object id in the `lopdf` page map.
///
/// `lopdf` keys its page map by `u32`, but callers hand us a `usize` page
/// number that comes from untrusted input. A truncating `page as u32` cast
/// wraps any value at or above `2^32` back into the low range — e.g. page
/// `2^32 + 1` narrows to `1` — so the lookup would silently return a *different*
/// page's object id instead of erroring. Range-checking the narrow with
/// `u32::try_from` turns that wrap into a typed [`PdfError::PageOutOfRange`],
/// and a legitimately-missing key maps to the same error.
fn page_object_id(
    pages_map: &std::collections::BTreeMap<u32, lopdf::ObjectId>,
    page: usize,
) -> Result<lopdf::ObjectId, PdfError> {
    let total = pages_map.len();
    let key = u32::try_from(page).map_err(|_| PdfError::PageOutOfRange { page, total })?;
    pages_map
        .get(&key)
        .copied()
        .ok_or(PdfError::PageOutOfRange { page, total })
}

/// The largest page count pdfium can represent.
///
/// pdfium-render 0.9 widened `PdfPageIndex` from `u16` to `c_int` (`i32`),
/// which is `FPDF_GetPageCount`'s own return width, so the old in-wrapper
/// `u16` truncation (#91) is gone by construction: any count pdfium can
/// report is representable as an index. What remains is the width of the C
/// API itself: `FPDF_GetPageCount` returns a `c_int`, so a document whose
/// true page count exceeds `i32::MAX` cannot be reported faithfully by
/// pdfium at all (pdfium-render clamps a negative count to `0`).
#[cfg_attr(not(feature = "pdfium"), allow(dead_code))]
const PDFIUM_MAX_PAGE_COUNT: usize = i32::MAX as usize;

/// Reject a document whose true page count exceeds pdfium's `c_int` page-count
/// width (see [`PDFIUM_MAX_PAGE_COUNT`]).
///
/// A count past `i32::MAX` cannot round-trip through `FPDF_GetPageCount`
/// (`c_int`), so by the time pdfium-render's pages accessor reports a length,
/// the true count is unrecoverable. This guard runs on the pdfium render paths
/// against the document's *true* page count (taken from `lopdf`, the same
/// authoritative page structure [`page_object_id`] trusts), so a misreported
/// count can never silently drive a page lookup.
#[cfg_attr(not(feature = "pdfium"), allow(dead_code))]
fn check_pdfium_page_count(true_count: usize) -> Result<(), PdfError> {
    if true_count > PDFIUM_MAX_PAGE_COUNT {
        return Err(PdfError::PageCountExceedsIndex {
            count: true_count,
            max: PDFIUM_MAX_PAGE_COUNT,
        });
    }
    Ok(())
}

/// Guard the pdfium render paths against a page count that overflows pdfium's
/// `c_int` page count (see [`check_pdfium_page_count`]).
///
/// The document's true page count is read from `lopdf`, whose page map is keyed
/// by `u32` and so can count pages past `i32::MAX` without pdfium's `c_int`
/// ceiling. A file that `lopdf` cannot parse is left untouched (pdfium may
/// still be able to render it), so this only rejects a document `lopdf` can
/// read whose page count exceeds the index width.
#[cfg(feature = "pdfium")]
fn reject_pages_beyond_pdfium_index(path: &Path) -> Result<(), PdfError> {
    if let Ok(doc) = lopdf::Document::load(path) {
        check_pdfium_page_count(doc.get_pages().len())?;
    }
    Ok(())
}

/// Resolve a 1-based `page` number to pdfium-render's 0-based
/// [`PdfPageIndex`](pdfium_render::prelude::PdfPageIndex) (`c_int`),
/// bounds-checked against the page count the document reports.
///
/// `total` comes from `PdfPages::len()`, which pdfium-render 0.9 clamps to
/// `0..=i32::MAX`, so a `page` that passes the bounds check always fits the
/// index type. The narrowing is still range-checked rather than cast so a
/// future change to the bounds check can never silently reintroduce a
/// wrapping conversion (#91).
#[cfg(feature = "pdfium")]
pub(crate) fn pdfium_page_index(
    page: usize,
    total: pdfium_render::prelude::PdfPageIndex,
) -> Result<pdfium_render::prelude::PdfPageIndex, PdfError> {
    // `PdfPages::len()` cannot return a negative count (pdfium-render clamps
    // it to 0), but saturate rather than assume: a negative count means no
    // page is addressable.
    let total = usize::try_from(total).unwrap_or(0);
    if page == 0 || page > total {
        return Err(PdfError::PageOutOfRange { page, total });
    }
    pdfium_render::prelude::PdfPageIndex::try_from(page - 1)
        .map_err(|_| PdfError::PageOutOfRange { page, total })
}

/// Extract the largest embedded raster image from a PDF page.
///
/// This is the fast path for scanned blueprints: the page typically contains
/// a single large JPEG or JPEG2000 image. We extract the raw compressed stream
/// and decode it with the `image` crate, avoiding any PDF rendering.
///
/// For vector PDFs that don't contain embedded images, use
/// [`render_page_pdfium`] instead (requires the `pdfium` feature).
///
/// # Examples
///
/// See [pdf_ops tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pdf_ops.rs),
/// [pdf_to_pyramid tests](https://github.com/libviprs/libviprs-tests/blob/main/tests/pdf_to_pyramid.rs),
/// and the [CLI pyramid command](https://github.com/libviprs/libviprs-cli/blob/main/src/main.rs).
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-page) (the
/// `--page` flag selects which page to extract; the [full pyramid flow](https://libviprs.org/cli/#pyramid)
/// uses this when the page has embedded images).
pub fn extract_page_image(path: &Path, page: usize) -> Result<Raster, PdfError> {
    let doc = lopdf::Document::load(path).map_err(|e| PdfError::Parse(e.to_string()))?;
    extract_page_image_from_doc(&doc, page)
}

/// Extract the largest embedded image from a page of an already-loaded
/// document. Shared by [`extract_page_image`] and
/// [`extract_page_image_with_password`] so the document is loaded once.
fn extract_page_image_from_doc(doc: &lopdf::Document, page: usize) -> Result<Raster, PdfError> {
    let pages_map = doc.get_pages();
    let page_id = page_object_id(&pages_map, page)?;

    extract_largest_image(doc, page_id, page)
}

/// Maximum permitted PDF image dimension, in pixels per axis.
///
/// Mirrors libvips' `VIPS_MAX_COORD` (10,000,000). `/Width` and `/Height`
/// come from untrusted PDF input; any value outside `1..=MAX_PDF_DIM` is
/// rejected before any arithmetic so that a wrapping `as u32`/`as usize`
/// cast or an overflowing size product can never escape into a `Raster`.
const MAX_PDF_DIM: i64 = 10_000_000;

/// Read and validate an image dimension (`/Width` or `/Height`) from a
/// stream dictionary.
///
/// Returns the value as `u32` only when it is present, integer-typed, and
/// within `1..=MAX_PDF_DIM`. A missing, non-integer, zero, negative, or
/// oversized value yields [`PdfError::UnsupportedFormat`] rather than a
/// wrapping cast that would defeat later size checks.
fn read_dimension(stream: &lopdf::Stream, key: &[u8]) -> Result<u32, PdfError> {
    let raw = stream
        .dict
        .get(key)
        .ok()
        .and_then(|v| v.as_i64().ok())
        .ok_or_else(|| {
            PdfError::UnsupportedFormat(format!(
                "missing or non-integer /{}",
                String::from_utf8_lossy(key)
            ))
        })?;
    if !(1..=MAX_PDF_DIM).contains(&raw) {
        return Err(PdfError::UnsupportedFormat(format!(
            "/{} = {raw} out of range 1..={MAX_PDF_DIM}",
            String::from_utf8_lossy(key)
        )));
    }
    Ok(raw as u32)
}

/// Extract the largest XObject Image from the page's resources.
fn extract_largest_image(
    doc: &lopdf::Document,
    page_id: lopdf::ObjectId,
    page: usize,
) -> Result<Raster, PdfError> {
    let page_obj = doc
        .get_object(page_id)
        .map_err(|e| PdfError::Parse(e.to_string()))?;

    // Get the page dictionary
    let page_dict = page_obj
        .as_dict()
        .map_err(|e| PdfError::Parse(e.to_string()))?;

    // Resolve Resources
    let resources = resolve_dict_entry(doc, page_dict, b"Resources")?;

    // Resolve XObject from Resources
    let xobjects = resolve_dict_entry(doc, resources, b"XObject")?;

    let mut best: Option<(usize, ImageData)> = None;

    for (_name, obj_ref) in xobjects.iter() {
        let obj_id = match obj_ref {
            lopdf::Object::Reference(id) => *id,
            _ => continue,
        };

        let obj = match doc.get_object(obj_id) {
            Ok(o) => o,
            Err(_) => continue,
        };

        let stream = match obj.as_stream() {
            Ok(s) => s,
            Err(_) => continue,
        };

        // Check it's an Image XObject
        let subtype = stream
            .dict
            .get(b"Subtype")
            .ok()
            .and_then(|s| s.as_name().ok());
        if subtype != Some(b"Image") {
            continue;
        }

        // Get image dimensions for size comparison. Validated dimensions are
        // each within `1..=MAX_PDF_DIM`, so their product fits comfortably in
        // `u64`; a malformed (missing/negative/oversized) dimension makes this
        // XObject ineligible for selection rather than tripping a wrapping
        // multiply.
        let (width, height) = match (
            read_dimension(stream, b"Width"),
            read_dimension(stream, b"Height"),
        ) {
            (Ok(w), Ok(h)) => (w, h),
            _ => continue,
        };
        let pixel_count = match (width as usize).checked_mul(height as usize) {
            Some(p) => p,
            None => continue,
        };

        // Get the image data (may be compressed with filters like DCTDecode/FlateDecode)
        let data = get_image_data(doc, stream)?;

        if best
            .as_ref()
            .is_none_or(|(best_size, _)| pixel_count > *best_size)
        {
            best = Some((pixel_count, data));
        }
    }

    let (_size, image_data) = best.ok_or(PdfError::NoImageOnPage { page })?;

    match image_data {
        ImageData::Decoded(raster) => Ok(raster),
        ImageData::Encoded(bytes) => {
            source::decode_bytes(&bytes).map_err(|e| PdfError::Decode(e.to_string()))
        }
    }
}

/// Decoded image data from a PDF stream — either encoded bytes (JPEG etc.)
/// that need further decoding, or an already-decoded Raster (from FlateDecode).
#[derive(Debug)]
enum ImageData {
    /// Encoded image bytes (JPEG, PNG, JPEG2000) — pass to `decode_bytes`.
    Encoded(Vec<u8>),
    /// Already-decoded raster (from raw FlateDecode pixel data).
    Decoded(Raster),
}

/// Resolve the filter(s) on a PDF stream into a list of filter names.
///
/// Handles both single-name filters (`/DCTDecode`) and filter arrays
/// (`[/FlateDecode /DCTDecode]`).
fn resolve_filters(stream: &lopdf::Stream) -> Vec<Vec<u8>> {
    let filter_obj = match stream.dict.get(b"Filter").ok() {
        Some(f) => f,
        None => return vec![],
    };

    // Single name filter
    if let Ok(name) = filter_obj.as_name() {
        return vec![name.to_vec()];
    }

    // Array of filters
    if let Ok(arr) = filter_obj.as_array() {
        return arr
            .iter()
            .filter_map(|f| f.as_name().ok().map(|n| n.to_vec()))
            .collect();
    }

    vec![]
}

/// Get image data from a PDF stream, handling common filters.
///
/// Supports single filters and chained filter arrays (e.g. `[/FlateDecode /DCTDecode]`).
fn get_image_data(doc: &lopdf::Document, stream: &lopdf::Stream) -> Result<ImageData, PdfError> {
    let filters = resolve_filters(stream);

    // Normalize: treat chained [FlateDecode, DCTDecode] as "decompress then JPEG"
    let terminal_filter: &[u8] = match filters.as_slice() {
        [] => b"",
        [single] => single,
        [first, second] if first.as_slice() == b"FlateDecode" => {
            // Chained: FlateDecode wrapping another format — decompress first,
            // then treat the inner data according to the second filter.
            let cap = max_decompressed_bytes(stream)?;
            let decompressed = flate_decompress(&stream.content, cap)?;
            return dispatch_single_filter(doc, stream, second, decompressed);
        }
        _ => {
            let names: Vec<String> = filters
                .iter()
                .map(|f| String::from_utf8_lossy(f).to_string())
                .collect();
            return Err(PdfError::UnsupportedFormat(format!(
                "filter chain: [{}]",
                names.join(", ")
            )));
        }
    };

    dispatch_single_filter(doc, stream, terminal_filter, stream.content.clone())
}

/// Handle a single filter applied to image data.
fn dispatch_single_filter(
    doc: &lopdf::Document,
    stream: &lopdf::Stream,
    filter: &[u8],
    data: Vec<u8>,
) -> Result<ImageData, PdfError> {
    match filter {
        b"DCTDecode" => {
            // JPEG data — return raw, let `image` crate decode
            Ok(ImageData::Encoded(data))
        }
        b"FlateDecode" => {
            // Deflate-compressed raw pixels
            let cap = max_decompressed_bytes(stream)?;
            let decompressed = flate_decompress(&data, cap)?;
            decode_raw_pixels(doc, stream, decompressed)
        }
        b"JPXDecode" => {
            // JPEG 2000 — return raw, let `image` crate attempt decode
            Ok(ImageData::Encoded(data))
        }
        b"" => {
            // No filter — try as encoded image, fall back to raw pixels
            Ok(ImageData::Encoded(data))
        }
        other => Err(PdfError::UnsupportedFormat(
            String::from_utf8_lossy(other).to_string(),
        )),
    }
}

/// Decode raw (uncompressed) pixel data using the stream's image metadata.
fn decode_raw_pixels(
    doc: &lopdf::Document,
    stream: &lopdf::Stream,
    decompressed: Vec<u8>,
) -> Result<ImageData, PdfError> {
    let width = read_dimension(stream, b"Width")?;
    let height = read_dimension(stream, b"Height")?;
    let bpc = stream
        .dict
        .get(b"BitsPerComponent")
        .ok()
        .and_then(|b| b.as_i64().ok())
        .unwrap_or(8) as u32;
    let cs = stream
        .dict
        .get(b"ColorSpace")
        .ok()
        .and_then(|c| resolve_object(doc, c).ok())
        .and_then(|c| c.as_name().ok().map(|n| n.to_vec()));

    let color_space: &[u8] = cs.as_deref().unwrap_or(b"DeviceRGB");

    let format = match (color_space, bpc) {
        (b"DeviceGray", 8) => PixelFormat::Gray8,
        (b"DeviceGray", 16) => PixelFormat::Gray16,
        (b"DeviceRGB", 8) => PixelFormat::Rgb8,
        (b"DeviceRGB", 16) => PixelFormat::Rgb16,
        (b"DeviceCMYK", _) => {
            let raster = cmyk_to_rgb_raster(&decompressed, width, height)?;
            return Ok(ImageData::Decoded(raster));
        }
        _ => {
            return Err(PdfError::UnsupportedFormat(format!(
                "{} @ {bpc}bpc",
                String::from_utf8_lossy(color_space)
            )));
        }
    };

    // Dimensions are already bounded by `MAX_PDF_DIM`, but compute the byte
    // count with checked arithmetic so the size guard can never be defeated by
    // a wrapped-small `expected` on any target width.
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|px| px.checked_mul(format.bytes_per_pixel()))
        .ok_or_else(|| {
            PdfError::UnsupportedFormat(format!(
                "image size {width}x{height} @ {} bpp overflows",
                format.bytes_per_pixel()
            ))
        })?;
    if decompressed.len() < expected {
        return Err(PdfError::Decode(format!(
            "decompressed size {} < expected {expected}",
            decompressed.len()
        )));
    }
    let mut data = decompressed;
    data.truncate(expected);

    let raster = Raster::new(width, height, format, data).map_err(PdfError::Raster)?;
    Ok(ImageData::Decoded(raster))
}

/// Widest per-pixel byte footprint we support decoding: CMYK (4 components)
/// at 16 bits-per-component = 8 bytes/pixel. Used to bound `/FlateDecode`
/// output relative to a stream's *declared* `/Width`×`/Height` so a crafted
/// stream cannot inflate past what its own dimensions could ever justify.
const MAX_BYTES_PER_PIXEL: usize = 8;

/// Fixed slack added to the declared-pixel bound to tolerate row/byte
/// alignment padding and encoder framing (e.g. a JPEG re-wrapped in
/// `[/FlateDecode /DCTDecode]`) without letting output grow without limit.
const FLATE_SLACK_BYTES: usize = 4096;

/// Compute the maximum permissible `/FlateDecode` output for `stream`, derived
/// from its validated `/Width`×`/Height` times the widest supported per-pixel
/// size, plus fixed slack.
///
/// The dimensions are validated by [`read_dimension`] (each within
/// `1..=MAX_PDF_DIM`) and the products are computed with checked arithmetic,
/// so a malformed or overflowing size yields a typed error instead of a
/// wrapped-small cap that a zip bomb could slip under.
fn max_decompressed_bytes(stream: &lopdf::Stream) -> Result<usize, PdfError> {
    let width = read_dimension(stream, b"Width")?;
    let height = read_dimension(stream, b"Height")?;
    (width as usize)
        .checked_mul(height as usize)
        .and_then(|px| px.checked_mul(MAX_BYTES_PER_PIXEL))
        .and_then(|bytes| bytes.checked_add(FLATE_SLACK_BYTES))
        .ok_or_else(|| {
            PdfError::UnsupportedFormat(format!(
                "decompression bound for {width}x{height} overflows"
            ))
        })
}

/// Decompress zlib/deflate data, capping the output at `max_output` bytes.
///
/// A `/FlateDecode` stream can inflate by orders of magnitude; without a bound
/// a few-KB stream can expand to gigabytes and OOM-abort via the infallible
/// allocation path (a zip bomb). The decoder is wrapped in a `Take` limited to
/// one byte past the cap so an over-limit stream is detected and rejected with
/// a typed [`PdfError::DecompressionLimitExceeded`] rather than decompressed in
/// full.
fn flate_decompress(data: &[u8], max_output: usize) -> Result<Vec<u8>, PdfError> {
    use std::io::Read;
    let decoder = flate2::read::ZlibDecoder::new(data);
    let mut out = Vec::new();
    // Read at most one byte beyond the cap: if that extra byte materialises the
    // stream exceeds the bound and is rejected below.
    decoder
        .take(max_output as u64 + 1)
        .read_to_end(&mut out)
        .map_err(|e| PdfError::Decode(format!("flate decompress: {e}")))?;
    if out.len() > max_output {
        return Err(PdfError::DecompressionLimitExceeded { limit: max_output });
    }
    Ok(out)
}

/// Convert CMYK raw bytes to RGB Raster.
fn cmyk_to_rgb_raster(cmyk_data: &[u8], width: u32, height: u32) -> Result<Raster, PdfError> {
    // `width`/`height` reach here already bounded by `MAX_PDF_DIM`, but keep
    // the input-size and output-capacity products checked so a malformed
    // stream can never wrap `pixel_count * 4` (guard) or `pixel_count * 3`
    // (allocation) into a small value.
    let pixel_count = width as usize * height as usize;
    let cmyk_len = pixel_count.checked_mul(4).ok_or_else(|| {
        PdfError::UnsupportedFormat(format!("CMYK size {width}x{height} overflows"))
    })?;
    let rgb_len = pixel_count.checked_mul(3).ok_or_else(|| {
        PdfError::UnsupportedFormat(format!("RGB size {width}x{height} overflows"))
    })?;
    if cmyk_data.len() < cmyk_len {
        return Err(PdfError::Decode("CMYK data too short".to_string()));
    }
    let mut rgb = Vec::with_capacity(rgb_len);
    for chunk in cmyk_data[..cmyk_len].as_chunks::<4>().0 {
        let c = chunk[0] as f32 / 255.0;
        let m = chunk[1] as f32 / 255.0;
        let y = chunk[2] as f32 / 255.0;
        let k = chunk[3] as f32 / 255.0;
        rgb.push(((1.0 - c) * (1.0 - k) * 255.0) as u8);
        rgb.push(((1.0 - m) * (1.0 - k) * 255.0) as u8);
        rgb.push(((1.0 - y) * (1.0 - k) * 255.0) as u8);
    }
    Raster::new(width, height, PixelFormat::Rgb8, rgb).map_err(PdfError::Raster)
}

/// Get page dimensions in points, in the *displayed* orientation.
///
/// Reads `/MediaBox` and applies the effective `/Rotate` from the page
/// dictionary (inheriting through the `/Parent` chain per PDF 1.7
/// §7.7.3.3). For `/Rotate 90` or `270`, width and height are swapped so
/// the returned `(w, h)` matches what viewers and the pdfium form-data
/// render path report. `/Rotate 0` and `180` preserve the MediaBox
/// orientation. Missing `/Rotate` behaves as `0`.
fn get_page_dimensions(doc: &lopdf::Document, page_id: lopdf::ObjectId) -> (f64, f64) {
    let obj = match doc.get_object(page_id) {
        Ok(o) => o,
        Err(_) => return (0.0, 0.0),
    };
    let dict = match obj.as_dict() {
        Ok(d) => d,
        Err(_) => return (0.0, 0.0),
    };

    // Try MediaBox, falling back through parent pages
    if let Some(media_box) = resolve_array_entry(doc, dict, b"MediaBox")
        && media_box.len() >= 4
    {
        let x0 = obj_to_f64(&media_box[0]).unwrap_or(0.0);
        let y0 = obj_to_f64(&media_box[1]).unwrap_or(0.0);
        let x1 = obj_to_f64(&media_box[2]).unwrap_or(0.0);
        let y1 = obj_to_f64(&media_box[3]).unwrap_or(0.0);
        let w = (x1 - x0).abs();
        let h = (y1 - y0).abs();
        return apply_rotate_to_dims(w, h, resolve_rotate(doc, page_id));
    }

    (0.0, 0.0)
}

/// Swap `(w, h)` when `rotate` is `90` or `270` (mod 360). Returns
/// `(w, h)` unchanged for `0`, `180`, or any non-multiple-of-90 value.
fn apply_rotate_to_dims(w: f64, h: f64, rotate: i64) -> (f64, f64) {
    // PDF spec: /Rotate must be a multiple of 90. `rem_euclid` normalises
    // negative or out-of-range values (seen in malformed PDFs) into
    // `[0, 360)` before the swap decision.
    let normalised = rotate.rem_euclid(360);
    if normalised == 90 || normalised == 270 {
        (h, w)
    } else {
        (w, h)
    }
}

/// Resolve the effective `/Rotate` value for a page.
///
/// Walks the `/Parent` chain per PDF 1.7 §7.7.3.3: the page's own dict
/// first, then each ancestor `Pages` node, returning the first numeric
/// `/Rotate` encountered. Returns `0` if none is found or the traversal
/// hits a malformed node / self-referential loop.
fn resolve_rotate(doc: &lopdf::Document, page_id: lopdf::ObjectId) -> i64 {
    let mut current = page_id;
    // Cap the walk defensively — page trees more than a few dozen levels
    // deep are pathological, and this stops an adversarial self-referential
    // `/Parent` from spinning forever even if the ID-equality guard below
    // misses it (e.g. alternating pairs of ids).
    for _ in 0..64 {
        let Ok(obj) = doc.get_object(current) else {
            return 0;
        };
        let Ok(dict) = obj.as_dict() else {
            return 0;
        };
        if let Ok(rotate_obj) = dict.get(b"Rotate")
            && let Ok(resolved) = resolve_object(doc, rotate_obj)
            && let Some(v) = obj_to_f64(resolved)
        {
            return v as i64;
        }
        // `/Parent` is an indirect reference to the parent Pages node.
        // Pull the object id straight from the Reference — don't call
        // `resolve_object`, which would hand back the dict itself and
        // lose the id we need to walk upward.
        let parent_id = match dict.get(b"Parent") {
            Ok(lopdf::Object::Reference(id)) => *id,
            _ => return 0,
        };
        if parent_id == current {
            return 0;
        }
        current = parent_id;
    }
    0
}

/// Resolve the effective `/Rotate` value for a 1-based page number.
///
/// Loads the PDF via `lopdf` and walks the page's `/Parent` chain to find
/// the inherited `/Rotate` entry (per PDF 1.7 §7.7.3.3). The result is
/// normalised into one of the four [`PageRotation`] variants. Pages
/// without a `/Rotate` entry, missing values, and self-referential
/// parent chains all resolve to [`PageRotation::Zero`].
///
/// This is the path-based companion of the crate-internal `resolve_rotate`
/// helper, which answers the same question for an already-open `lopdf`
/// document. Callers driving pdfium's matrix render path need the page's
/// intrinsic `/Rotate` to compose the right device transform —
/// `FPDF_RenderPageBitmapWithMatrix` does not auto-apply it the way the
/// form-data render path does.
///
/// # Errors
///
/// - [`PdfError::Parse`] — PDF could not be opened or parsed by `lopdf`.
/// - [`PdfError::PageOutOfRange`] — `page == 0` or `page > total_pages`.
/// - [`PdfError::UnsupportedRotation`] — the resolved `/Rotate` value
///   is not a multiple of 90 (PDF spec violation).
pub fn page_rotate(path: &Path, page: usize) -> Result<PageRotation, PdfError> {
    let doc = lopdf::Document::load(path).map_err(|e| PdfError::Parse(e.to_string()))?;
    let pages_map = doc.get_pages();
    let page_id = page_object_id(&pages_map, page)?;
    PageRotation::try_from_degrees(resolve_rotate(&doc, page_id))
}

/// Check whether a page has any Image XObjects.
fn page_has_images(doc: &lopdf::Document, page_id: lopdf::ObjectId) -> bool {
    let Ok(obj) = doc.get_object(page_id) else {
        return false;
    };
    let Ok(dict) = obj.as_dict() else {
        return false;
    };
    let Ok(resources) = resolve_dict_entry(doc, dict, b"Resources") else {
        return false;
    };
    let Ok(xobjects) = resolve_dict_entry(doc, resources, b"XObject") else {
        return false;
    };

    for (_name, obj_ref) in xobjects.iter() {
        let obj_id = match obj_ref {
            lopdf::Object::Reference(id) => *id,
            _ => continue,
        };
        let Ok(obj) = doc.get_object(obj_id) else {
            continue;
        };
        let Ok(stream) = obj.as_stream() else {
            continue;
        };
        let subtype = stream
            .dict
            .get(b"Subtype")
            .ok()
            .and_then(|s| s.as_name().ok());
        if subtype == Some(b"Image") {
            return true;
        }
    }

    false
}

// -- lopdf helpers --

fn resolve_dict_entry<'a>(
    doc: &'a lopdf::Document,
    dict: &'a lopdf::Dictionary,
    key: &[u8],
) -> Result<&'a lopdf::Dictionary, PdfError> {
    let entry = dict
        .get(key)
        .map_err(|_| PdfError::Parse(format!("missing key: {}", String::from_utf8_lossy(key))))?;
    let resolved = resolve_object(doc, entry)?;
    resolved
        .as_dict()
        .map_err(|_| PdfError::Parse(format!("{} is not a dict", String::from_utf8_lossy(key))))
}

fn resolve_array_entry<'a>(
    doc: &'a lopdf::Document,
    dict: &'a lopdf::Dictionary,
    key: &[u8],
) -> Option<&'a Vec<lopdf::Object>> {
    let entry = dict.get(key).ok()?;
    let resolved = resolve_object(doc, entry).ok()?;
    resolved.as_array().ok()
}

fn resolve_object<'a>(
    doc: &'a lopdf::Document,
    obj: &'a lopdf::Object,
) -> Result<&'a lopdf::Object, PdfError> {
    match obj {
        lopdf::Object::Reference(id) => doc
            .get_object(*id)
            .map_err(|e| PdfError::Parse(e.to_string())),
        other => Ok(other),
    }
}

fn obj_to_f64(obj: &lopdf::Object) -> Option<f64> {
    match obj {
        lopdf::Object::Integer(i) => Some(*i as f64),
        lopdf::Object::Real(f) => Some(*f as f64),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Pdfium-based rendering (feature-gated)
// ---------------------------------------------------------------------------

/// Process-wide lock for FPDF API access — defence-in-depth thread
/// safety at the libviprs boundary.
///
/// pdfium itself is not thread-safe at the C library level. The
/// `pdfium-render` `sync` feature wraps every FPDF call in a global
/// mutex, but **only** in the per-call locking fork that libviprs
/// declares as a direct git dependency. Consumers that build libviprs
/// from git or a path inherit that edge, but the crates.io-published
/// libviprs cannot carry a git source (cargo strips it on publish),
/// so registry consumers link the upstream wrapper whose locking is
/// broken (issue #149; upstream ajrcarey/pdfium-render#262).
///
/// To keep libviprs correct without depending on which wrapper the
/// consumer's build resolved, every FPDF entry point in this crate
/// acquires this process-wide lock first. If the fork is also active
/// (the recommended setup), the result is double-locking: a few extra
/// nanoseconds per call, no correctness or deadlock concern (the
/// locks are independent `Mutex<()>` instances acquired in a fixed
/// order). If the fork is missing, this lock alone keeps concurrent
/// renders safe.
///
/// **Performance note:** With this lock held, multi-threaded
/// `render_strip` calls serialise. That matches the underlying
/// reality (pdfium itself is single-threaded), so no parallelism is
/// lost — this lock simply makes the serialisation explicit and safe
/// instead of implicit and crashing.
#[cfg(feature = "pdfium")]
static PDFIUM_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Lock a `Mutex<()>` used purely for serialisation, recovering from
/// poison instead of propagating it. If a thread panicked while holding
/// the lock the mutex is poisoned; every one of these locks guards
/// nothing but access to the single-threaded pdfium C library, and a
/// Rust panic on the Rust side cannot corrupt that library's internal
/// state, so the poison is benign and must not brick later callers.
/// Both [`pdfium_lock`] and the init serialisation guard in
/// [`init_pdfium`] go through here so the recovery is uniform.
#[cfg_attr(not(feature = "pdfium"), allow(dead_code))]
#[inline]
fn lock_recovering(mutex: &std::sync::Mutex<()>) -> std::sync::MutexGuard<'_, ()> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Acquire [`PDFIUM_LOCK`]. Panics in another thread that previously
/// held the lock are recovered from rather than propagated; pdfium is
/// a C library and a Rust panic on the Rust side cannot corrupt its
/// internal state, so poisoning here is benign.
#[cfg(feature = "pdfium")]
#[inline]
pub(crate) fn pdfium_lock() -> std::sync::MutexGuard<'static, ()> {
    lock_recovering(&PDFIUM_LOCK)
}

/// Decide the explicit pdfium library path from a raw `PDFIUM_PATH` value.
///
/// Pure policy helper so the resolution decision is unit-testable without
/// touching the process environment or an installed pdfium binary.
///
/// - `None` / empty → the caller falls back to the system library search
///   path. The current working directory is **never** substituted, closing
///   the CWE-427 library-injection vector where a planted
///   `./libpdfium.{dylib,so,dll}` in a writable working directory would be
///   loaded ahead of the trusted system library.
/// - a directory → the platform library file name is appended by the
///   caller via `pdfium_platform_library_name_at_path`.
/// - a file → used verbatim as the library to bind.
#[cfg_attr(not(feature = "pdfium"), allow(dead_code))]
fn resolve_pdfium_path(raw: Option<std::ffi::OsString>) -> Option<std::path::PathBuf> {
    let raw = raw?;
    if raw.is_empty() {
        return None;
    }
    Some(std::path::PathBuf::from(raw))
}

/// Open a pdfium instance with the appropriate bindings.
#[cfg(feature = "pdfium")]
pub(crate) fn init_pdfium() -> Result<&'static pdfium_render::prelude::Pdfium, PdfError> {
    use pdfium_render::prelude::*;
    use std::sync::{Mutex, OnceLock};

    // Pdfium's FPDF_InitLibrary must be called exactly once per process —
    // calling it twice while a prior instance is alive deadlocks inside the
    // C library on macOS. Keep a single process-wide instance behind a
    // OnceLock; the init path is serialised by INIT_GUARD so concurrent first
    // callers can't both invoke FPDF_InitLibrary.
    static PDFIUM: OnceLock<Pdfium> = OnceLock::new();
    static INIT_GUARD: Mutex<()> = Mutex::new(());

    if let Some(p) = PDFIUM.get() {
        return Ok(p);
    }
    // Recover from poison rather than `.unwrap()`: if a prior init attempt
    // panicked while holding this guard (e.g. inside `bind_to_system_library`
    // or `Pdfium::new`) the mutex is poisoned, and propagating that would make
    // every future `init_pdfium` panic here, permanently disabling all PDF
    // rendering in the process (#89). The guard only serialises init, so the
    // poison is benign.
    let _guard = lock_recovering(&INIT_GUARD);
    if let Some(p) = PDFIUM.get() {
        return Ok(p);
    }
    #[cfg(feature = "pdfium-static")]
    let bindings =
        Pdfium::bind_to_statically_linked_library().map_err(|e| PdfError::Pdfium(e.to_string()))?;
    // Resolve the library explicitly from `PDFIUM_PATH`, then fall back to
    // the trusted system search path. The current working directory is never
    // consulted (CWE-427): a directory value has the platform library file
    // name appended, a file value is bound verbatim, and an unset/empty value
    // defers entirely to `bind_to_system_library`.
    #[cfg(not(feature = "pdfium-static"))]
    let bindings = match resolve_pdfium_path(std::env::var_os("PDFIUM_PATH")) {
        Some(path) => {
            let library = if path.is_dir() {
                Pdfium::pdfium_platform_library_name_at_path(&path)
            } else {
                path
            };
            Pdfium::bind_to_library(library).map_err(|e| PdfError::Pdfium(e.to_string()))?
        }
        None => Pdfium::bind_to_system_library().map_err(|e| PdfError::Pdfium(e.to_string()))?,
    };
    let pdfium = Pdfium::new(bindings);
    let _ = PDFIUM.set(pdfium);
    Ok(PDFIUM.get().expect("PDFIUM was just set"))
}

/// Convert a rendered pdfium bitmap into an RGBA [`Raster`].
///
/// `as_image` is the bitmap conversion closure (in production,
/// `|| bitmap.as_image()`). pdfium-render 0.9's `as_image()` returns
/// `Result`: a width/height that disagrees with the buffer length (the
/// case that made the 0.8.x `as_image()` panic through its terminal
/// `from_raw(..).unwrap()`) now surfaces as
/// `Err(PdfiumError::ImageError)`, which this maps to a typed
/// [`PdfError::Pdfium`].
///
/// The [`std::panic::catch_unwind`] isolation predates that upstream fix
/// and is kept as a backstop: the closure still crosses FFI-adjacent
/// wrapper code, and a panic reaching a MapReduce render worker aborts
/// the rendering thread. Any residual panic is converted into the same
/// typed [`PdfError::Pdfium`] the caller can propagate.
#[cfg(feature = "pdfium")]
fn bitmap_to_raster<F>(as_image: F) -> Result<Raster, PdfError>
where
    F: FnOnce() -> Result<image::DynamicImage, pdfium_render::prelude::PdfiumError>,
{
    let img = std::panic::catch_unwind(std::panic::AssertUnwindSafe(as_image))
        .map_err(|_| {
            PdfError::Pdfium(
                "panic during pdfium bitmap conversion (PdfBitmap::as_image)".to_string(),
            )
        })?
        .map_err(|e| PdfError::Pdfium(e.to_string()))?;
    let rgba = img.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());
    let data = rgba.into_raw();
    Raster::new(w, h, PixelFormat::Rgba8, data).map_err(PdfError::from)
}

/// Bytes pdfium's BGRA bitmap buffer will span for a `width`x`height` render.
///
/// pdfium's `FPDFBitmap_GetBuffer_as_vec` computes the buffer length as an
/// `i32 * i32` product (`stride * height`) before casting to `usize`
/// (libviprs/pdfium-render `bindings.rs:3127-3129`). For a large enough bitmap
/// that product overflows `i32::MAX`, wrapping to a negative value
/// (sign-extended to a ~1.8e19-byte out-of-bounds slice) or to a small
/// positive value (silent raster truncation). This guard rejects such a render
/// on the libviprs side before we ever hand pdfium those dimensions.
///
/// `stride` is pdfium's tightest 32-bit BGRA row length, `width * 4` bytes;
/// real pdfium strides are `>=` this, so the check is conservative.
///
/// A zero `width` or `height` is also refused: a page whose dimensions round
/// to 0 px at a low DPI would otherwise reach pdfium and yield a degenerate
/// bitmap that divides by zero in `as_rgba_bytes` (`bytes.len() / height`).
/// See #86.
#[cfg_attr(not(feature = "pdfium"), allow(dead_code))]
fn pdfium_bitmap_span(width: u32, height: u32) -> Result<usize, PdfError> {
    // Compute in u64 so the multiply itself cannot overflow, then reject any
    // span pdfium's i32 buffer-length arithmetic could not represent. A zero
    // width yields a zero/negative stride, and a zero height yields a
    // zero-row bitmap whose stride pdfium reports as the buffer length —
    // `as_rgba_bytes` then divides `bytes.len() / height` and panics with a
    // divide-by-zero. Both degenerate dimensions are equally unusable, so
    // reject a page that rounds to 0 px in either axis before pdfium ever
    // sees it (#86).
    let stride = u64::from(width) * 4;
    let span = stride * u64::from(height);
    if width == 0 || height == 0 || span > i32::MAX as u64 {
        return Err(PdfError::RenderTooLarge {
            width,
            height,
            span,
        });
    }
    Ok(span as usize)
}

/// Render a page at the given pixel dimensions and return a Raster.
#[cfg(feature = "pdfium")]
pub(crate) fn render_at_size(
    pdf_page: &pdfium_render::prelude::PdfPage<'_>,
    width: u32,
    height: u32,
) -> Result<Raster, PdfError> {
    use pdfium_render::prelude::*;

    // Refuse renders whose bitmap buffer would overflow pdfium's i32 span (#148).
    pdfium_bitmap_span(width, height)?;

    let config = PdfRenderConfig::new()
        .set_target_width(width as i32)
        .set_maximum_height(height as i32);

    let bitmap = pdf_page
        .render_with_config(&config)
        .map_err(|e| PdfError::Pdfium(e.to_string()))?;

    bitmap_to_raster(|| bitmap.as_image())
}

/// Build pdfium's clear colour from a `background` colour slice.
///
/// The slice is read as `[r, g, b]` or `[r, g, b, a]` with each channel a
/// `0..=255` intensity; a missing alpha defaults to fully opaque. A missing or
/// non-finite channel (NaN or infinity) falls back to its default before
/// clamping, so a NaN alpha reads as opaque (its default) rather than
/// saturating to a transparent `0`, and out-of-range values clamp into
/// `0..=255`. Narrowing to pdfium's `u8` channels can therefore never wrap.
/// This stays total (the public entry point validates the channel count up
/// front), so a short slice can never panic here.
#[cfg(feature = "pdfium")]
fn background_to_pdf_color(background: &[f64]) -> pdfium_render::prelude::PdfColor {
    let channel = |index: usize, default: f64| -> u8 {
        let raw = background.get(index).copied().unwrap_or(default);
        let value = if raw.is_finite() { raw } else { default };
        value.clamp(0.0, 255.0).round() as u8
    };
    pdfium_render::prelude::PdfColor::new(
        channel(0, 0.0),
        channel(1, 0.0),
        channel(2, 0.0),
        channel(3, 255.0),
    )
}

/// Render a page at the given pixel dimensions over a solid `background` fill
/// and return a Raster.
///
/// Identical to [`render_at_size`] except the render config clears the output
/// bitmap to `background` before drawing the page (pdfium's
/// [`clear_before_rendering`](pdfium_render::prelude::PdfRenderConfig::clear_before_rendering)
/// over [`set_clear_color`](pdfium_render::prelude::PdfRenderConfig::set_clear_color),
/// whose default is
/// [`PdfColor::WHITE`](pdfium_render::prelude::PdfColor::WHITE)). Any region
/// the page leaves transparent then shows `background` instead of the default
/// white. `background` is read as `[r, g, b]` or `[r, g, b, a]`; see
/// [`background_to_pdf_color`] for the per-channel handling.
#[cfg(feature = "pdfium")]
pub(crate) fn render_at_size_with_background(
    pdf_page: &pdfium_render::prelude::PdfPage<'_>,
    width: u32,
    height: u32,
    background: &[f64],
) -> Result<Raster, PdfError> {
    use pdfium_render::prelude::*;

    // Refuse renders whose bitmap buffer would overflow pdfium's i32 span (#148).
    pdfium_bitmap_span(width, height)?;

    let config = PdfRenderConfig::new()
        .set_target_width(width as i32)
        .set_maximum_height(height as i32)
        .set_clear_color(background_to_pdf_color(background))
        .clear_before_rendering(true);

    let bitmap = pdf_page
        .render_with_config(&config)
        .map_err(|e| PdfError::Pdfium(e.to_string()))?;

    bitmap_to_raster(|| bitmap.as_image())
}

/// Render a PDF page to a raster using PDFium.
///
/// This handles vector content (AutoCAD exports, text, paths) that cannot be
/// extracted as embedded images. Requires the `pdfium` feature and a PDFium
/// library available at runtime.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-render)
#[cfg(feature = "pdfium")]
pub fn render_page_pdfium(path: &Path, page: usize, dpi: u32) -> Result<Raster, PdfError> {
    reject_pages_beyond_pdfium_index(path)?;
    let pdfium = init_pdfium()?;
    let _lock = pdfium_lock();
    let document = pdfium
        .load_pdf_from_file(path, None)
        .map_err(|e| PdfError::Pdfium(e.to_string()))?;

    let pages = document.pages();
    let index = pdfium_page_index(page, pages.len())?;
    let pdf_page = pages
        .get(index)
        .map_err(|e| PdfError::Pdfium(e.to_string()))?;

    let (width, height) = render_dims_within_budget(
        pdf_page.width().value,
        pdf_page.height().value,
        dpi,
        DEFAULT_MAX_RENDER_PIXELS,
    )?;

    render_at_size(&pdf_page, width, height)
}

/// Render a PDF page to a raster over a solid `background` fill using pdfium.
///
/// Mirrors [`render_page_pdfium`] but clears the output bitmap to `background`
/// before drawing, so any region the page leaves transparent shows the
/// requested colour instead of the default white. See
/// [`render_at_size_with_background`] for the clear-colour handling. This backs
/// [`extract_page_image_with_background`].
#[cfg(feature = "pdfium")]
fn render_page_pdfium_with_background(
    path: &Path,
    page: usize,
    dpi: u32,
    background: &[f64],
) -> Result<Raster, PdfError> {
    reject_pages_beyond_pdfium_index(path)?;
    let pdfium = init_pdfium()?;
    let _lock = pdfium_lock();
    let document = pdfium
        .load_pdf_from_file(path, None)
        .map_err(|e| PdfError::Pdfium(e.to_string()))?;

    let pages = document.pages();
    let index = pdfium_page_index(page, pages.len())?;
    let pdf_page = pages
        .get(index)
        .map_err(|e| PdfError::Pdfium(e.to_string()))?;

    let (width, height) = render_dims_within_budget(
        pdf_page.width().value,
        pdf_page.height().value,
        dpi,
        DEFAULT_MAX_RENDER_PIXELS,
    )?;

    render_at_size_with_background(&pdf_page, width, height, background)
}

/// Compose the device matrix passed to `FPDF_RenderPageBitmapWithMatrix`
/// for a strip starting at `y_offset` (display pixels down the page),
/// rendered at `scale = dpi / 72.0`.
///
/// The caller matrix is **not** where rotation is applied. pdfium's matrix
/// render path composes this matrix on top of the page's own display matrix,
/// which already applies the page's intrinsic `/Rotate`, flips y-up PDF
/// points to y-down device pixels, and maps into the destination rect. So
/// by the time this matrix runs, the page is already in **device space**:
/// top-left origin, y-down, correctly rotated, `1 pt · scale = 1 px`.
///
/// A strip therefore needs only a uniform `scale` and a downward translation
/// of `-y_offset`, so display rows `[y_offset, y_offset + strip_h)` land at
/// the top of the strip-sized bitmap. Rotation and the y-flip are pdfium's
/// job. Baking a per-`/Rotate` rotation in here (as this function used to)
/// double-applies the rotation and transposes/clips rotated pages — see the
/// `rotation_libvips_pdfium_parity` regression test in libviprs-tests.
///
/// Returns `[a, b, c, d, e, f] = [scale, 0, 0, scale, 0, -y_offset]`.
#[cfg(feature = "pdfium")]
#[must_use]
pub(crate) fn strip_matrix(scale: f32, y_offset: u32) -> [f32; 6] {
    [scale, 0.0, 0.0, scale, 0.0, -(y_offset as f32)]
}

/// Render a single horizontal strip of a PDF page directly via pdfium's
/// matrix render path, allocating only a strip-sized bitmap.
///
/// `y_offset` and `strip_height` are in display-oriented pixel coordinates
/// (top-left origin, y-down) — the same coordinate system the engine and
/// `StripSource` callers already speak.
///
/// # Coordinate composition
///
/// `FPDF_RenderPageBitmapWithMatrix` composes the caller matrix on top of
/// the page's own display matrix, which already applies the page's intrinsic
/// `/Rotate`, flips y-up PDF points to y-down device pixels, and scales into
/// the destination rect. The caller matrix therefore operates in device
/// space, so a strip needs only a uniform scale plus a `-y_offset`
/// translation to bring display rows `[y_offset, y_offset + strip_height)` to
/// rows `[0, strip_height)` of the output bitmap. Rotation is pdfium's job,
/// not ours — see [`strip_matrix`]. Page dimensions come from
/// `pdf_page.width()/height()`, which already return display-oriented
/// (post-`/Rotate`) values, so no dimension swap is needed here. The rendered
/// pixels are pinned against libvips output for every `/Rotate` value by
/// libviprs-tests' `rotation_libvips_pdfium_parity` test.
///
/// # Errors
///
/// - [`PdfError::PageOutOfRange`] — `page == 0` or `page > total_pages`.
/// - [`PdfError::Pdfium`] — pdfium load / page get / matrix-validity /
///   render error.
///
/// Render a single horizontal strip from an **already-loaded** [`PdfPage`].
///
/// This is the hot-path entry used by [`crate::PdfiumStripSource`] in
/// streaming mode: callers cache the parsed `PdfDocument` / `PdfPage`
/// once at construction and reuse them across every `render_strip`
/// call, avoiding the per-strip PDF reparse that path-based one-shot
/// rendering would pay.
///
/// FPDF calls underneath this function are serialised by
/// `pdfium-render`'s `ThreadSafePdfiumBindings` (active via the
/// `sync` feature plus the direct git dependency on the per-call
/// locking fork in `libviprs/Cargo.toml`).
///
/// `dpi`, `y_offset`, `strip_height` semantics match the device matrix
/// documented at [`strip_matrix`].
#[cfg(feature = "pdfium")]
pub(crate) fn render_page_strip_with_page(
    pdf_page: &pdfium_render::prelude::PdfPage<'_>,
    dpi: u32,
    y_offset: u32,
    strip_height: u32,
) -> Result<Raster, PdfError> {
    use pdfium_render::prelude::*;

    let scale = dpi as f32 / 72.0;
    // Display-oriented dims; pdf_page.width()/height() return post-/Rotate
    // values per `pdf.rs:376-379`.
    let display_w_pt = pdf_page.width().value;
    let display_h_pt = pdf_page.height().value;
    let display_w_px = (display_w_pt * scale) as u32;
    let display_h_px = (display_h_pt * scale) as u32;

    // Mirror cached-mode clamping (streaming.rs:341-346): if the requested
    // strip extends past the page, the bitmap is shorter; if it starts
    // past the page, return a zero raster of the requested height to
    // keep the StripSource contract.
    let strip_h = strip_height.min(display_h_px.saturating_sub(y_offset));
    if strip_h == 0 {
        let data = alloc_zeroed_rgba(display_w_px, strip_height)?;
        return Raster::new(display_w_px, strip_height, PixelFormat::Rgba8, data)
            .map_err(PdfError::from);
    }

    let [a, b, c, d, e, f] = strip_matrix(scale, y_offset);
    let matrix = PdfMatrix::new(a, b, c, d, e, f);

    let config = PdfRenderConfig::new()
        .set_fixed_size(display_w_px as i32, strip_h as i32)
        .clip(0, 0, display_w_px as i32, strip_h as i32)
        .apply_matrix(matrix)
        .map_err(|e| PdfError::Pdfium(e.to_string()))?;

    let bitmap = pdf_page
        .render_with_config(&config)
        .map_err(|e| PdfError::Pdfium(e.to_string()))?;

    bitmap_to_raster(|| bitmap.as_image())
}

/// Result of a budget-constrained render, including the DPI that was used.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-render)
#[cfg(feature = "pdfium")]
#[derive(Debug)]
pub struct BudgetRenderResult {
    pub raster: Raster,
    pub dpi_used: u32,
    pub capped: bool,
}

/// Render a PDF page to a raster with a memory safety net.
///
/// Unlike [`render_page_pdfium`] which renders at exactly the requested DPI
/// regardless of output size, this function caps the total pixel count to
/// prevent OOM when rendering large-format PDFs (e.g. a 48"x36" AutoCAD
/// blueprint at 300 DPI = 518 megapixels). It picks whichever constraint
/// is more restrictive — the requested DPI or the pixel budget — and
/// reduces DPI automatically if needed.
///
/// Use [`render_page_pdfium`] when you control the DPI and know the output
/// will fit in memory. Use this function in pipelines where the PDF page
/// size is unknown and you need a memory ceiling.
///
/// - `max_dpi`: the preferred DPI (e.g. 300). Used when the result fits
///   within the budget.
/// - `max_pixels`: maximum total pixel count (width * height). If rendering
///   at `max_dpi` would exceed this, the DPI is automatically reduced so
///   the output fits.
///
/// Returns the raster along with the actual DPI used and whether it was capped.
///
/// **See also:** [interactive example](https://libviprs.org/cli/#flag-render)
#[cfg(feature = "pdfium")]
pub fn render_page_pdfium_budgeted(
    path: &Path,
    page: usize,
    max_dpi: u32,
    max_pixels: u64,
) -> Result<BudgetRenderResult, PdfError> {
    reject_pages_beyond_pdfium_index(path)?;
    let pdfium = init_pdfium()?;
    let _lock = pdfium_lock();
    let document = pdfium
        .load_pdf_from_file(path, None)
        .map_err(|e| PdfError::Pdfium(e.to_string()))?;

    let pages = document.pages();
    let index = pdfium_page_index(page, pages.len())?;
    let pdf_page = pages
        .get(index)
        .map_err(|e| PdfError::Pdfium(e.to_string()))?;

    let width_pts = pdf_page.width().value as f64;
    let height_pts = pdf_page.height().value as f64;

    // Compute the DPI that fits within the pixel budget.
    let (dpi_used, capped) = budgeted_render_dpi(width_pts, height_pts, max_dpi, max_pixels);

    let scale = dpi_used as f32 / 72.0;
    let width = (width_pts as f32 * scale) as u32;
    let height = (height_pts as f32 * scale) as u32;

    let raster = render_at_size(&pdf_page, width, height)?;

    Ok(BudgetRenderResult {
        raster,
        dpi_used,
        capped,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Save a `lopdf` document to a fresh temp file and return the directory
    /// (kept alive for the file's lifetime) alongside the path.
    fn save_doc(mut doc: lopdf::Document) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("doc.pdf");
        doc.save(&path).unwrap();
        (dir, path)
    }

    /// A minimal, readable, unencrypted 1-page PDF on disk.
    fn save_plain_pdf() -> (tempfile::TempDir, std::path::PathBuf) {
        let (doc, _page_id) = build_rotated_doc([0.0, 0.0, 100.0, 100.0], None, None);
        save_doc(doc)
    }

    /// A 1-page PDF carrying a standard-security `/Encrypt` dictionary, so the
    /// reloaded document reports `is_encrypted()`. Per the PDF spec the
    /// encryption dictionary itself is not encrypted, so no crypto is needed to
    /// make the file parse and read back as encrypted.
    fn save_encrypted_pdf() -> (tempfile::TempDir, std::path::PathBuf) {
        use lopdf::{Object, dictionary};
        let (mut doc, _page_id) = build_rotated_doc([0.0, 0.0, 100.0, 100.0], None, None);
        let encrypt_id = doc.add_object(dictionary! {
            "Filter" => "Standard",
            "V" => 1i64,
            "R" => 2i64,
            "P" => -1i64,
        });
        doc.trailer.set("Encrypt", Object::Reference(encrypt_id));
        save_doc(doc)
    }

    #[test]
    fn pdf_info_with_password_reports_encrypted_as_unsupported() {
        // A genuinely encrypted document (one carrying an /Encrypt dictionary)
        // opened with a non-empty password reports the typed
        // unsupported-capability error, since this build cannot decrypt it.
        let (_dir, path) = save_encrypted_pdf();
        match pdf_info_with_password(&path, "secret") {
            Err(PdfError::UnsupportedFormat(msg)) => {
                assert!(msg.contains("password"), "message was {msg:?}");
            }
            other => panic!("expected UnsupportedFormat, got {other:?}"),
        }
    }

    #[test]
    fn pdf_info_with_password_passes_through_open_error() {
        // A missing/unreadable file must surface its real open error, not the
        // password-protected message, even when a password is supplied.
        let bogus = Path::new("/nonexistent/secret.pdf");
        match pdf_info_with_password(bogus, "secret") {
            Err(PdfError::Parse(msg)) => {
                assert!(!msg.contains("password"), "message was {msg:?}");
            }
            other => panic!("expected the underlying open error, got {other:?}"),
        }
        // An empty password behaves identically.
        assert!(pdf_info_with_password(bogus, "").is_err());
    }

    #[test]
    fn pdf_info_with_password_ignores_password_on_unencrypted_doc() {
        // A normal, readable document is not mislabelled as password-protected
        // just because a password was supplied.
        let (_dir, path) = save_plain_pdf();
        let info = pdf_info_with_password(&path, "secret").expect("plain doc reads back");
        assert_eq!(info.page_count, 1);
    }

    #[test]
    fn password_and_dpi_extract_return_a_clean_typed_error() {
        let bogus = Path::new("/nonexistent/secret.pdf");
        // A missing file surfaces its real error folded into the decode error,
        // not the password-protected capability message.
        let err = extract_page_image_with_password(bogus, 1, "secret").unwrap_err();
        assert!(
            !err.to_string().contains("password-protected"),
            "missing file was mislabelled: {err}"
        );
        assert!(extract_page_image_dpi(bogus, 1, 72.0).is_err());
    }

    #[test]
    fn extract_page_image_with_password_reports_encrypted_as_unsupported() {
        // An encrypted document opened with a password folds to the typed
        // password-protected capability error.
        let (_dir, path) = save_encrypted_pdf();
        let err = extract_page_image_with_password(&path, 1, "secret").unwrap_err();
        assert!(
            err.to_string().contains("password-protected"),
            "encrypted doc should report password-protected, got: {err}"
        );
    }

    /// Read one RGBA pixel out of a `PixelFormat::Rgba8` raster's row-major
    /// byte buffer. Test-only, so an out-of-range coordinate panics.
    #[cfg(feature = "pdfium")]
    fn rgba_pixel_at(raster: &Raster, x: u32, y: u32) -> [u8; 4] {
        assert_eq!(raster.format(), PixelFormat::Rgba8);
        let idx = ((y * raster.width() + x) * 4) as usize;
        let px = &raster.data()[idx..idx + 4];
        [px[0], px[1], px[2], px[3]]
    }

    /// The background render honours its clear colour: a page whose only
    /// painted content is a small square in one corner renders the rest of the
    /// bitmap in the requested colour. Rendering with a white background and
    /// with a red background both succeed, yield equal dimensions, and differ
    /// pixel-for-pixel in the transparent region (see #87). Gated on `pdfium`
    /// because the assertion performs a real page render; without the feature
    /// the function reports an unsupported-capability decode error instead.
    #[cfg(feature = "pdfium")]
    #[test]
    fn extract_page_image_with_background_applies_clear_colour() {
        // A 100x100pt page whose sole content is a 10x10pt black square in the
        // bottom-left corner. Every other region is unpainted, so pdfium's
        // clear colour shows through it. That makes it a genuinely transparent
        // fixture, the precondition a background pixel-difference assertion needs.
        let (doc, _page_id) = build_rotated_doc([0.0, 0.0, 100.0, 100.0], None, None);
        let (_dir, path) = save_doc(doc);

        let white = extract_page_image_with_background(&path, 1, &[255.0, 255.0, 255.0])
            .expect("white-background render succeeds");
        let red = extract_page_image_with_background(&path, 1, &[255.0, 0.0, 0.0])
            .expect("red-background render succeeds");

        // Same page at the same default DPI: identical dimensions.
        assert!(white.width() > 0 && white.height() > 0);
        assert_eq!(
            (white.width(), white.height()),
            (red.width(), red.height()),
            "both backgrounds must render the page at the same size"
        );

        // Sample the top-right corner (far from the bottom-left square, so it
        // sits in the transparent region). The clear colour reaches it: white
        // for the white background, red for the red background.
        let (cx, cy) = (white.width() - 1, 0);
        let white_corner = rgba_pixel_at(&white, cx, cy);
        let red_corner = rgba_pixel_at(&red, cx, cy);
        assert_eq!(
            white_corner,
            [255, 255, 255, 255],
            "white background must fill the transparent region with white"
        );
        assert_eq!(
            red_corner,
            [255, 0, 0, 255],
            "red background must fill the transparent region with red"
        );
        assert_ne!(
            white_corner, red_corner,
            "different backgrounds must produce different transparent-region pixels"
        );
    }

    /// A non-finite channel falls back to its default rather than saturating
    /// through the `u8` cast: a NaN alpha reads as opaque (its `255` default),
    /// non-finite colour channels read as their `0` default, and finite
    /// out-of-range values still clamp into `0..=255`. Guards the non-finite
    /// contract of [`background_to_pdf_color`] (see #87).
    #[cfg(feature = "pdfium")]
    #[test]
    fn background_to_pdf_color_maps_non_finite_channels_to_defaults() {
        // A NaN alpha must land on the opaque default, not a transparent 0.
        let nan_alpha = background_to_pdf_color(&[10.0, 20.0, 30.0, f64::NAN]);
        assert_eq!(
            (
                nan_alpha.red(),
                nan_alpha.green(),
                nan_alpha.blue(),
                nan_alpha.alpha()
            ),
            (10, 20, 30, 255),
            "a NaN alpha must fall back to the opaque default"
        );

        // Every non-finite variant (NaN, +inf, -inf) resolves to its channel
        // default rather than wrapping the cast.
        let non_finite =
            background_to_pdf_color(&[f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::NAN]);
        assert_eq!(
            (
                non_finite.red(),
                non_finite.green(),
                non_finite.blue(),
                non_finite.alpha()
            ),
            (0, 0, 0, 255),
            "non-finite channels fall back to their defaults"
        );

        // Finite out-of-range values still clamp into 0..=255.
        let clamped = background_to_pdf_color(&[-5.0, 300.0, 128.0]);
        assert_eq!(
            (
                clamped.red(),
                clamped.green(),
                clamped.blue(),
                clamped.alpha()
            ),
            (0, 255, 128, 255),
            "finite out-of-range values clamp into 0..=255"
        );
    }

    /// The entry point rejects a background slice with more than four channels
    /// instead of silently ignoring the trailing data: a 5-channel slice fails
    /// loudly with a typed invalid-input decode error (see #87).
    #[cfg(feature = "pdfium")]
    #[test]
    fn extract_page_image_with_background_rejects_over_length_slice() {
        let (doc, _page_id) = build_rotated_doc([0.0, 0.0, 100.0, 100.0], None, None);
        let (_dir, path) = save_doc(doc);

        let err = extract_page_image_with_background(&path, 1, &[0.0, 0.0, 0.0, 255.0, 0.0])
            .expect_err("a 5-channel background must be rejected");
        assert!(
            err.to_string()
                .contains("3 (r, g, b) or 4 (r, g, b, a) channels"),
            "over-length background must report the channel-count error, got: {err}"
        );
    }

    /// [`BackgroundColor`]'s [`From`] conversions pick the variant that matches
    /// the array's channel count: a `[u8; 3]` is opaque
    /// [`BackgroundColor::Rgb`] and a `[u8; 4]` is [`BackgroundColor::Rgba`].
    /// Runs without `pdfium` since it exercises only the typed conversion, not
    /// a render (see #323).
    #[test]
    fn background_color_from_arrays_picks_channel_count() {
        assert_eq!(
            BackgroundColor::from([1u8, 2, 3]),
            BackgroundColor::Rgb([1, 2, 3])
        );
        assert_eq!(
            BackgroundColor::from([1u8, 2, 3, 4]),
            BackgroundColor::Rgba([1, 2, 3, 4])
        );
        // The generic `impl Into<BackgroundColor>` bound the typed entry point
        // uses accepts the same array literals.
        fn take(c: impl Into<BackgroundColor>) -> BackgroundColor {
            c.into()
        }
        assert_eq!(take([9u8, 8, 7]), BackgroundColor::Rgb([9, 8, 7]));
        assert_eq!(take([9u8, 8, 7, 6]), BackgroundColor::Rgba([9, 8, 7, 6]));
    }

    /// The typed entry point [`extract_page_image_with_background_typed`]
    /// produces a byte-identical render to the loosely-typed
    /// [`extract_page_image_with_background`] for an equivalent colour: it
    /// delegates to the `&[f64]` form with the same channels, so both the RGB
    /// and RGBA variants match dimensions, format, and pixel data. Gated on
    /// `pdfium` because it performs a real page render (see #323); without the
    /// feature both entry points report an unsupported-capability decode error.
    #[cfg(feature = "pdfium")]
    #[test]
    fn typed_background_matches_slice_form() {
        let (doc, _page_id) = build_rotated_doc([0.0, 0.0, 100.0, 100.0], None, None);
        let (_dir, path) = save_doc(doc);

        // Byte-for-byte raster equality: dimensions, pixel format, and data.
        fn assert_same(a: &Raster, b: &Raster) {
            assert_eq!((a.width(), a.height()), (b.width(), b.height()));
            assert_eq!(a.format(), b.format());
            assert_eq!(a.data(), b.data());
        }

        // 3-channel opaque RGB: typed `[u8; 3]` == slice `&[f64; 3]`.
        let typed_rgb = extract_page_image_with_background_typed(&path, 1, [200u8, 40, 10])
            .expect("typed RGB render succeeds");
        let slice_rgb = extract_page_image_with_background(&path, 1, &[200.0, 40.0, 10.0])
            .expect("slice RGB render succeeds");
        assert_same(&typed_rgb, &slice_rgb);

        // 4-channel RGBA: typed `[u8; 4]` == slice `&[f64; 4]`.
        let typed_rgba = extract_page_image_with_background_typed(&path, 1, [200u8, 40, 10, 128])
            .expect("typed RGBA render succeeds");
        let slice_rgba = extract_page_image_with_background(&path, 1, &[200.0, 40.0, 10.0, 128.0])
            .expect("slice RGBA render succeeds");
        assert_same(&typed_rgba, &slice_rgba);
    }

    /// Regression for #91: `page_object_id` must range-check the `usize -> u32`
    /// page-number narrowing. A page number at or above `2^32` truncates under
    /// the old `page as u32` cast — e.g. `2^32 + 1` wraps to `1` — which would
    /// silently return page 1's object id instead of erroring. The guarded
    /// `u32::try_from` narrow must surface a typed [`PdfError::PageOutOfRange`].
    #[cfg(target_pointer_width = "64")]
    #[test]
    fn page_object_id_rejects_index_wrapping_u32() {
        use std::collections::BTreeMap;

        let mut pages_map: BTreeMap<u32, lopdf::ObjectId> = BTreeMap::new();
        pages_map.insert(1, (1, 0));

        // 2^32 + 1 narrows to 1 under a truncating `as u32` cast, which would
        // wrongly resolve to page 1's object id (1, 0).
        let page = (1usize << 32) + 1;
        assert_eq!(page as u32, 1, "precondition: the cast wraps to page 1");

        let err = page_object_id(&pages_map, page).unwrap_err();
        assert!(
            matches!(err, PdfError::PageOutOfRange { page: p, total: 1 } if p == page),
            "wrapped page index must be rejected, got {err:?}"
        );

        // Sanity: an in-range page still resolves to its object id.
        assert_eq!(page_object_id(&pages_map, 1).unwrap(), (1, 0));
    }

    /// A page number that fits in `u32` but is not present in the map still
    /// yields [`PdfError::PageOutOfRange`] — the guarded narrow must not change
    /// the behaviour for an ordinary out-of-range lookup.
    #[test]
    fn page_object_id_rejects_missing_in_range_page() {
        use std::collections::BTreeMap;

        let mut pages_map: BTreeMap<u32, lopdf::ObjectId> = BTreeMap::new();
        pages_map.insert(1, (1, 0));

        let err = page_object_id(&pages_map, 2).unwrap_err();
        assert!(
            matches!(err, PdfError::PageOutOfRange { page: 2, total: 1 }),
            "an in-range but absent page must be rejected, got {err:?}"
        );
    }

    /// Follow-up to #91: pdfium-render 0.9 widened `PdfPageIndex` from `u16`
    /// to `c_int` (`i32`), so the in-wrapper `u16` truncation the original
    /// guard targeted is gone by construction. The residual limit is the C
    /// API's own width: `FPDF_GetPageCount` returns a `c_int`, so a true page
    /// count past `i32::MAX` cannot be reported faithfully.
    /// `check_pdfium_page_count` must reject such a count with a typed
    /// [`PdfError::PageCountExceedsIndex`] instead of letting a misreported
    /// count silently drive a page lookup.
    #[test]
    fn check_pdfium_page_count_rejects_index_width_overflow() {
        // One page past the c_int width is unrepresentable by
        // `FPDF_GetPageCount`'s return type: the exact overflow this guards.
        let overflowing = PDFIUM_MAX_PAGE_COUNT + 1;
        assert!(
            i32::try_from(overflowing).is_err(),
            "precondition: the count exceeds what c_int can represent"
        );

        let err = check_pdfium_page_count(overflowing).unwrap_err();
        assert!(
            matches!(
                err,
                PdfError::PageCountExceedsIndex { count, max }
                    if count == overflowing && max == PDFIUM_MAX_PAGE_COUNT
            ),
            "a page count past the c_int width must be rejected, got {err:?}"
        );
    }

    /// The boundary count (`i32::MAX`) and any smaller count are representable
    /// by pdfium's page count and must be accepted unchanged: the guard must
    /// not reject documents pdfium can address.
    #[test]
    fn check_pdfium_page_count_accepts_up_to_index_width() {
        assert!(check_pdfium_page_count(0).is_ok());
        assert!(check_pdfium_page_count(1).is_ok());
        assert!(check_pdfium_page_count(PDFIUM_MAX_PAGE_COUNT).is_ok());
    }

    /// `pdfium_page_index` is the single 1-based-page to 0-based-`c_int`
    /// conversion every pdfium render path uses (follow-up to #91, migrated
    /// to pdfium-render 0.9's `i32` `PdfPageIndex`). It must bounds-check
    /// before narrowing: page 0 and pages past the reported total map to
    /// [`PdfError::PageOutOfRange`], in-range pages map to `page - 1`, and a
    /// (theoretically impossible) negative total addresses no page at all.
    #[cfg(feature = "pdfium")]
    #[test]
    fn pdfium_page_index_bounds_checks_before_narrowing() {
        // In-range pages convert to their 0-based index.
        assert_eq!(pdfium_page_index(1, 3).unwrap(), 0);
        assert_eq!(pdfium_page_index(3, 3).unwrap(), 2);
        assert_eq!(
            pdfium_page_index(i32::MAX as usize, i32::MAX).unwrap(),
            i32::MAX - 1
        );

        // Page 0 (pages are 1-based) and pages past the total are rejected.
        assert!(matches!(
            pdfium_page_index(0, 3).unwrap_err(),
            PdfError::PageOutOfRange { page: 0, total: 3 }
        ));
        assert!(matches!(
            pdfium_page_index(4, 3).unwrap_err(),
            PdfError::PageOutOfRange { page: 4, total: 3 }
        ));

        // A negative reported total (pdfium-render clamps this to 0, but the
        // helper must not assume) addresses no page.
        assert!(matches!(
            pdfium_page_index(1, -1).unwrap_err(),
            PdfError::PageOutOfRange { page: 1, total: 0 }
        ));
    }

    /*
     * Regression for #89: the init serialisation guard must recover from
     * poison instead of permanently bricking pdfium initialisation.
     *
     * `init_pdfium` locks its `INIT_GUARD` through `lock_recovering`. If a
     * prior init attempt panicked while holding that guard (e.g. inside
     * `Pdfium::bind_to_system_library` or `Pdfium::new`) the mutex is
     * poisoned. A plain `.lock().unwrap()` would then propagate the poison to
     * every subsequent caller, so all future PDF rendering in the process
     * would die at that unwrap. `lock_recovering` — the single routine both
     * `init_pdfium` and `pdfium_lock` use — must hand back the guard instead.
     *
     * This exercises `lock_recovering` directly (init itself needs a bound
     * pdfium library, unavailable in a unit test) against a mutex poisoned the
     * exact way init would poison it: a thread panicking while holding it.
     * Before the fix the init path called `.unwrap()` and this recovery did
     * not exist, so the poisoned lock panicked; after the fix it recovers.
     */
    #[test]
    fn init_guard_recovers_from_poison() {
        use std::sync::{Arc, Mutex};

        let mutex: Arc<Mutex<()>> = Arc::new(Mutex::new(()));

        // Poison it the way a panic during init would: a thread panics while
        // holding the lock.
        let poisoner = Arc::clone(&mutex);
        let _ = std::thread::spawn(move || {
            let _held = poisoner.lock().unwrap();
            panic!("simulated panic during pdfium init");
        })
        .join();

        assert!(
            mutex.is_poisoned(),
            "precondition: the init guard must be poisoned"
        );

        // The routine the init path uses must recover rather than propagate
        // the poison; a `.unwrap()` here would panic and permanently disable
        // pdfium init.
        {
            let _guard = lock_recovering(&mutex);
        }

        // And it keeps working on the next attempt — init is not bricked.
        let _guard = lock_recovering(&mutex);
    }

    /**
     * Regression test for #148: pdfium's bitmap buffer length is computed as
     * an `i32 * i32` product (`stride * height`) which overflows `i32::MAX`
     * for large high-DPI pages, feeding an out-of-bounds `from_raw_parts`.
     *
     * Reproduces the issue's exact failure scenario — a 48"x36" blueprint at
     * 600 DPI: width 28800, stride 115200, height 21600, span 2_488_320_000
     * bytes (> i32::MAX = 2_147_483_647). `pdfium_bitmap_span` must return the
     * correct span computed in a wide integer, and must reject a zero-width
     * render, rather than overflowing.
     */
    #[test]
    fn pdfium_bitmap_span_no_i32_overflow() {
        // A large-but-representable render still succeeds, with the span
        // computed without overflow: 20000 * 4 * 20000 = 1_600_000_000 < i32::MAX.
        assert_eq!(
            pdfium_bitmap_span(20000, 20000).unwrap(),
            20000usize * 4 * 20000
        );

        // The blueprint case — 28800 * 4 * 21600 = 2_488_320_000 > i32::MAX —
        // is refused instead of overflowing pdfium's i32 buffer length and
        // feeding an out-of-bounds from_raw_parts read.
        assert!(matches!(
            pdfium_bitmap_span(28800, 21600),
            Err(PdfError::RenderTooLarge {
                span: 2_488_320_000,
                ..
            })
        ));

        // Zero width would give pdfium a zero/negative stride.
        assert!(matches!(
            pdfium_bitmap_span(0, 21600),
            Err(PdfError::RenderTooLarge { .. })
        ));
    }

    /*
     * Regression for #86: a page whose height rounds to 0 px must not reach
     * pdfium. `as_rgba_bytes` divides `bytes.len() / height`, so a zero-height
     * bitmap panics with a divide-by-zero. `render_at_size` calls
     * `pdfium_bitmap_span` first, so guarding a zero height there closes the
     * panic for both `render_page_pdfium` and `render_page_pdfium_budgeted`.
     *
     * Before the fix `pdfium_bitmap_span(100, 0)` returned `Ok(0)` (the guard
     * only rejected `width == 0`), letting the degenerate render proceed; after
     * the fix it returns a typed error.
     */
    #[test]
    fn pdfium_bitmap_span_rejects_zero_height() {
        // The exact scenario from #86: dimensions derived for a page that
        // rounds to 0 px tall at a low DPI (e.g. a 0.5 pt-tall page at 72 DPI).
        let (width, height) =
            render_dims_within_budget(100.0, 0.5, 72, DEFAULT_MAX_RENDER_PIXELS).unwrap();
        assert_eq!(height, 0, "a 0.5 pt page at 72 DPI must round to 0 px tall");
        assert!(
            width > 0,
            "width stays non-zero, isolating the height==0 case"
        );

        // Those dimensions must be refused with a typed error rather than
        // handed to pdfium (which would panic downstream in as_rgba_bytes).
        assert!(matches!(
            pdfium_bitmap_span(width, height),
            Err(PdfError::RenderTooLarge {
                height: 0,
                span: 0,
                ..
            })
        ));

        // A non-degenerate height on the same width still succeeds.
        assert!(pdfium_bitmap_span(width, 1).is_ok());
    }

    /**
     * Tests that CMYK-to-RGB conversion produces correct color values.
     * Works by converting a single pure-cyan pixel (C=255, M=0, Y=0, K=0)
     * and checking that the resulting RGB raster has R=0, G=255, B=255.
     * Input: 1x1 CMYK pixel [255, 0, 0, 0].
     * Output: 1x1 Rgb8 raster with data [0, 255, 255].
     */
    #[test]
    fn cmyk_to_rgb_basic() {
        // Pure cyan: C=255, M=0, Y=0, K=0 → R=0, G=255, B=255
        let cmyk = vec![255, 0, 0, 0];
        let raster = cmyk_to_rgb_raster(&cmyk, 1, 1).unwrap();
        assert_eq!(raster.width(), 1);
        assert_eq!(raster.height(), 1);
        assert_eq!(raster.format(), PixelFormat::Rgb8);
        let data = raster.data();
        assert_eq!(data[0], 0); // R
        assert_eq!(data[1], 255); // G
        assert_eq!(data[2], 255); // B
    }

    /**
     * Tests that obj_to_f64 correctly converts a lopdf Integer to f64.
     * Works by creating an Integer(42) object and verifying it returns Some(42.0),
     * confirming the integer-to-float promotion path.
     * Input: lopdf::Object::Integer(42). Output: Some(42.0).
     */
    /// Build a minimal raw-pixel image stream dictionary with the given
    /// `/Width` and `/Height` (as raw `i64`, so out-of-range and negative
    /// values can be injected) using `DeviceRGB` @ 16bpc and `FlateDecode`.
    fn crafted_image_stream(width: i64, height: i64) -> lopdf::Stream {
        use lopdf::{Stream, dictionary};
        Stream::new(
            dictionary! {
                "Type" => "XObject",
                "Subtype" => "Image",
                "Width" => width,
                "Height" => height,
                "BitsPerComponent" => 16i64,
                "ColorSpace" => "DeviceRGB",
                "Filter" => "FlateDecode",
            },
            Vec::new(),
        )
    }

    /// A `/Width u32::MAX /Height u32::MAX` `Rgb16` image must be rejected with
    /// a typed error before any size product is computed. On the pre-fix code
    /// `expected = width * height * bpp` overflows and panics in debug (wraps
    /// in release, letting a bogus `Raster` escape); after the fix the
    /// out-of-range dimension is rejected up front.
    #[test]
    fn decode_raw_pixels_rejects_oversized_dims() {
        let doc = lopdf::Document::with_version("1.5");
        let stream = crafted_image_stream(u32::MAX as i64, u32::MAX as i64);
        let err = decode_raw_pixels(&doc, &stream, vec![0u8; 16]).unwrap_err();
        assert!(
            matches!(err, PdfError::UnsupportedFormat(_)),
            "expected UnsupportedFormat, got {err:?}"
        );
    }

    /// A negative `/Width` (which the old `as u32`/`as usize` casts wrapped
    /// into an ~1.8e19 value) must be rejected as a typed error, not wrapped.
    #[test]
    fn decode_raw_pixels_rejects_negative_dims() {
        let doc = lopdf::Document::with_version("1.5");
        let stream = crafted_image_stream(-1, 16);
        let err = decode_raw_pixels(&doc, &stream, vec![0u8; 16]).unwrap_err();
        assert!(
            matches!(err, PdfError::UnsupportedFormat(_)),
            "expected UnsupportedFormat, got {err:?}"
        );
    }

    /// A crafted `/MediaBox` at a high DPI drives the derived pixel count into
    /// the billions; the non-budgeted render path must reject it with a typed
    /// [`PdfError::RenderBudgetExceeded`] before allocating, not proceed toward
    /// a multi-gigabyte pdfium bitmap + `Raster` and OOM-abort. Values mirror a
    /// ~200k × 200k pt page at 300 DPI (~695 billion px), which the saturating
    /// `f32 as u32` cast keeps large rather than wrapping small.
    #[test]
    fn render_dims_within_budget_rejects_oversized_page() {
        let err = render_dims_within_budget(200_000.0, 200_000.0, 300, DEFAULT_MAX_RENDER_PIXELS)
            .unwrap_err();
        assert!(
            matches!(err, PdfError::RenderBudgetExceeded { .. }),
            "expected RenderBudgetExceeded, got {err:?}"
        );
    }

    /// The exact-budget boundary is admitted (`<= max_pixels`), but one pixel
    /// past it is rejected — the ceiling is inclusive of the budget itself.
    #[test]
    fn render_dims_within_budget_boundary_is_inclusive() {
        // 32768 × 32768 = 2^30 px exactly, at 72 DPI (scale = 1.0).
        let ok = render_dims_within_budget(32_768.0, 32_768.0, 72, DEFAULT_MAX_RENDER_PIXELS);
        assert_eq!(ok.unwrap(), (32_768, 32_768));

        let over = render_dims_within_budget(32_768.0, 32_769.0, 72, DEFAULT_MAX_RENDER_PIXELS);
        assert!(
            matches!(over, Err(PdfError::RenderBudgetExceeded { .. })),
            "one pixel past the budget must be rejected, got {over:?}"
        );
    }

    /// A normal Letter page at a normal DPI decodes to the expected pixel
    /// dimensions — the ceiling must not reject legitimate input.
    #[test]
    fn render_dims_within_budget_accepts_normal_page() {
        // US Letter, 612 × 792 pt at 150 DPI.
        let (w, h) =
            render_dims_within_budget(612.0, 792.0, 150, DEFAULT_MAX_RENDER_PIXELS).unwrap();
        // Matches the production `(pts * scale) as u32` truncation byte-for-byte.
        assert_eq!((w, h), (1275, 1649));
    }

    /// An adversarial page whose pixel estimate overflows a `u64` product
    /// must still trip the DPI-reduction branch. Estimated in `u64`, the
    /// product `2^62 × 2^34 = 2^96` wraps to `0` and slips under the budget,
    /// bypassing the cap and rendering at full DPI. Estimated in `f64` the
    /// product is `~7.9e28`, far above the `2^30` budget, so the render must
    /// be capped.
    #[test]
    fn budgeted_render_dpi_overflow_cannot_bypass_budget() {
        // At max_dpi = 72 the scale is 1.0, so the device-pixel dimensions
        // equal the point dimensions: 2^62 × 2^34 device px.
        let width_pts = (1u64 << 62) as f64;
        let height_pts = (1u64 << 34) as f64;
        let (dpi_used, capped) =
            budgeted_render_dpi(width_pts, height_pts, 72, DEFAULT_MAX_RENDER_PIXELS);
        assert!(
            capped,
            "overflowing pixel estimate must still trigger DPI reduction"
        );
        assert!(
            dpi_used < 72,
            "capped DPI must be below the requested max, got {dpi_used}"
        );
    }

    /// A normal Letter page at 300 DPI (~2550 × 3300 ≈ 8.4M px) sits well
    /// under the `2^30` budget and must render at the requested DPI with no
    /// capping — the overflow-safe estimate must not over-reduce legitimate
    /// input.
    #[test]
    fn budgeted_render_dpi_normal_page_uses_max_dpi() {
        let (dpi_used, capped) = budgeted_render_dpi(612.0, 792.0, 300, DEFAULT_MAX_RENDER_PIXELS);
        assert_eq!(dpi_used, 300);
        assert!(!capped, "a page within budget must not be capped");
    }

    /// The zero-strip fill (`width × height × 4`) overflows `usize` for
    /// adversarial `u32` dimensions; the fallible allocator must surface a
    /// typed error rather than panic on the overflowing multiply or abort in
    /// the infallible `vec![0u8; ..]` path.
    #[test]
    fn alloc_zeroed_rgba_rejects_overflowing_dims() {
        let err = alloc_zeroed_rgba(u32::MAX, u32::MAX).unwrap_err();
        assert!(
            matches!(
                err,
                PdfError::RenderBudgetExceeded { .. } | PdfError::AllocationFailed { .. }
            ),
            "expected a typed error, got {err:?}"
        );
    }

    /// A small in-bounds request yields a correctly sized, zero-filled buffer.
    #[test]
    fn alloc_zeroed_rgba_allocates_small_buffer() {
        let buf = alloc_zeroed_rgba(2, 3).unwrap();
        assert_eq!(buf.len(), 2 * 3 * 4);
        assert!(buf.iter().all(|&b| b == 0));
    }

    /// A well-formed small raw-pixel image still decodes into a `Raster` with
    /// the declared dimensions — the hardening must not reject valid input.
    #[test]
    fn decode_raw_pixels_accepts_valid_small_image() {
        use lopdf::{Stream, dictionary};
        let doc = lopdf::Document::with_version("1.5");
        let stream = Stream::new(
            dictionary! {
                "Type" => "XObject",
                "Subtype" => "Image",
                "Width" => 2i64,
                "Height" => 1i64,
                "BitsPerComponent" => 8i64,
                "ColorSpace" => "DeviceRGB",
            },
            Vec::new(),
        );
        // 2x1 Rgb8 = 6 bytes.
        let data = vec![10, 20, 30, 40, 50, 60];
        let decoded = decode_raw_pixels(&doc, &stream, data).unwrap();
        match decoded {
            ImageData::Decoded(raster) => {
                assert_eq!(raster.width(), 2);
                assert_eq!(raster.height(), 1);
                assert_eq!(raster.format(), PixelFormat::Rgb8);
            }
            ImageData::Encoded(_) => panic!("expected decoded raster"),
        }
    }

    /// A dimension/length mismatch from pdfium (the case where 0.8.x's
    /// `PdfBitmap::as_image()` panicked through its terminal
    /// `from_raw(..).unwrap()`) surfaces in pdfium-render 0.9 as
    /// `Err(PdfiumError::ImageError)`. `bitmap_to_raster` must map that
    /// to a typed [`PdfError::Pdfium`], the same contract the old panic
    /// guard provided. The closure reproduces 0.9's exact terminal
    /// expression: a 2x2 RGBA image needs 16 bytes; it is handed 3, so
    /// `from_raw` returns `None` and `as_image` reports `ImageError`.
    #[cfg(feature = "pdfium")]
    #[test]
    fn bitmap_to_raster_surfaces_dimension_mismatch_as_error() {
        use image::{DynamicImage, RgbaImage};
        use pdfium_render::prelude::PdfiumError;
        let result = bitmap_to_raster(|| {
            RgbaImage::from_raw(2, 2, vec![0u8; 3])
                .map(DynamicImage::ImageRgba8)
                .ok_or(PdfiumError::ImageError)
        });
        match result {
            Err(PdfError::Pdfium(_)) => {}
            other => panic!("expected Err(PdfError::Pdfium), got {other:?}"),
        }
    }

    /// A panic escaping the conversion closure (the backstop case: 0.9's
    /// `as_image` returns `Result`, but the closure still crosses
    /// FFI-adjacent wrapper code) must be caught and surfaced as a typed
    /// [`PdfError::Pdfium`], not unwind out of the render worker and
    /// abort a MapReduce thread.
    #[cfg(feature = "pdfium")]
    #[test]
    fn bitmap_to_raster_surfaces_panic_as_error() {
        let result = bitmap_to_raster(|| panic!("simulated panic inside bitmap conversion"));
        match result {
            Err(PdfError::Pdfium(msg)) => {
                assert!(
                    msg.contains("panic"),
                    "message should name the panic: {msg}"
                );
            }
            other => panic!("expected Err(PdfError::Pdfium), got {other:?}"),
        }
    }

    /// A well-formed bitmap must still convert into a `Raster` with the
    /// declared dimensions: the error mapping must not reject valid input.
    #[cfg(feature = "pdfium")]
    #[test]
    fn bitmap_to_raster_accepts_valid_image() {
        use image::{DynamicImage, RgbaImage};
        let raster = bitmap_to_raster(|| {
            Ok(DynamicImage::ImageRgba8(
                RgbaImage::from_raw(2, 2, vec![255u8; 16]).unwrap(),
            ))
        })
        .unwrap();
        assert_eq!(raster.width(), 2);
        assert_eq!(raster.height(), 2);
        assert_eq!(raster.format(), PixelFormat::Rgba8);
    }

    #[test]
    fn obj_to_f64_integer() {
        let obj = lopdf::Object::Integer(42);
        assert_eq!(obj_to_f64(&obj), Some(42.0));
    }

    /**
     * Tests that obj_to_f64 correctly passes through a lopdf Real value.
     * Works by creating a Real(3.14) object and checking the returned f64
     * is within floating-point tolerance of 2.78.
     * Input: lopdf::Object::Real(2.78). Output: Some(~2.78).
     */
    #[test]
    fn obj_to_f64_real() {
        let obj = lopdf::Object::Real(2.78);
        assert!((obj_to_f64(&obj).unwrap() - 2.78).abs() < 0.001);
    }

    /**
     * Tests that obj_to_f64 returns None for non-numeric PDF object types.
     * Works by passing a Boolean object, which has no meaningful f64 conversion,
     * and verifying the function correctly rejects it with None.
     * Input: lopdf::Object::Boolean(true). Output: None.
     */
    #[test]
    fn obj_to_f64_other() {
        let obj = lopdf::Object::Boolean(true);
        assert_eq!(obj_to_f64(&obj), None);
    }

    // -----------------------------------------------------------------
    // /Rotate handling (issue #50)
    //
    // These tests pin the contract that `get_page_dimensions` returns
    // display-oriented dimensions: a landscape MediaBox with /Rotate 90
    // is reported as portrait, /Rotate 0 and /Rotate 180 preserve the
    // raw MediaBox orientation, and /Rotate is inheritable through the
    // Pages tree. Each test builds a minimal in-memory PDF via lopdf
    // so the contract is exercised end-to-end through the same parser
    // path `pdf_info` uses at runtime.
    // -----------------------------------------------------------------

    /// Build a minimal 1-page document with the given MediaBox +
    /// `page_rotate` on the leaf page and `pages_rotate` on the parent
    /// Pages node. `None` for either means the key is absent.
    fn build_rotated_doc(
        media_box: [f64; 4],
        page_rotate: Option<i64>,
        pages_rotate: Option<i64>,
    ) -> (lopdf::Document, lopdf::ObjectId) {
        use lopdf::{Document, Object, Stream, dictionary};

        let mut doc = Document::with_version("1.5");
        let pages_id = doc.new_object_id();
        let content = Stream::new(dictionary! {}, b"q 0 g 0 0 10 10 re f Q".to_vec());
        let content_id = doc.add_object(content);

        let mut page_dict = dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "MediaBox" => vec![
                media_box[0].into(),
                media_box[1].into(),
                media_box[2].into(),
                media_box[3].into(),
            ],
            "Contents" => content_id,
            "Resources" => dictionary! {},
        };
        if let Some(r) = page_rotate {
            page_dict.set("Rotate", Object::Integer(r));
        }
        let page_id = doc.add_object(page_dict);

        let mut pages_dict = dictionary! {
            "Type" => "Pages",
            "Kids" => vec![page_id.into()],
            "Count" => 1i64,
        };
        if let Some(r) = pages_rotate {
            pages_dict.set("Rotate", Object::Integer(r));
        }
        doc.objects.insert(pages_id, Object::Dictionary(pages_dict));

        let catalog_id = doc.add_object(dictionary! {
            "Type" => "Catalog",
            "Pages" => pages_id,
        });
        doc.trailer.set("Root", catalog_id);
        (doc, page_id)
    }

    #[test]
    fn get_page_dimensions_no_rotate_is_identity() {
        let (doc, page_id) = build_rotated_doc([0.0, 0.0, 1000.0, 500.0], None, None);
        assert_eq!(get_page_dimensions(&doc, page_id), (1000.0, 500.0));
    }

    #[test]
    fn get_page_dimensions_rotate_zero_is_identity() {
        let (doc, page_id) = build_rotated_doc([0.0, 0.0, 1000.0, 500.0], Some(0), None);
        assert_eq!(get_page_dimensions(&doc, page_id), (1000.0, 500.0));
    }

    #[test]
    fn get_page_dimensions_rotate_90_swaps() {
        let (doc, page_id) = build_rotated_doc([0.0, 0.0, 1000.0, 500.0], Some(90), None);
        assert_eq!(get_page_dimensions(&doc, page_id), (500.0, 1000.0));
    }

    #[test]
    fn get_page_dimensions_rotate_180_preserves() {
        let (doc, page_id) = build_rotated_doc([0.0, 0.0, 1000.0, 500.0], Some(180), None);
        assert_eq!(get_page_dimensions(&doc, page_id), (1000.0, 500.0));
    }

    #[test]
    fn get_page_dimensions_rotate_270_swaps() {
        let (doc, page_id) = build_rotated_doc([0.0, 0.0, 1000.0, 500.0], Some(270), None);
        assert_eq!(get_page_dimensions(&doc, page_id), (500.0, 1000.0));
    }

    #[test]
    fn get_page_dimensions_negative_rotate_normalises() {
        // /Rotate -90 is equivalent to 270 under rem_euclid normalisation.
        let (doc, page_id) = build_rotated_doc([0.0, 0.0, 1000.0, 500.0], Some(-90), None);
        assert_eq!(get_page_dimensions(&doc, page_id), (500.0, 1000.0));
    }

    #[test]
    fn get_page_dimensions_rotate_inherited_from_parent_pages() {
        // /Rotate on the parent Pages node, absent on the page itself.
        // PDF 1.7 §7.7.3.3 mandates inheritance; the page must pick up
        // the parent's rotation.
        let (doc, page_id) = build_rotated_doc([0.0, 0.0, 1000.0, 500.0], None, Some(90));
        assert_eq!(get_page_dimensions(&doc, page_id), (500.0, 1000.0));
    }

    #[test]
    fn get_page_dimensions_page_rotate_overrides_parent() {
        // When both the page and the parent Pages node declare /Rotate,
        // the page's value wins — the walk stops at the first /Rotate it
        // encounters.
        let (doc, page_id) = build_rotated_doc([0.0, 0.0, 1000.0, 500.0], Some(0), Some(90));
        assert_eq!(get_page_dimensions(&doc, page_id), (1000.0, 500.0));
    }

    #[test]
    fn apply_rotate_to_dims_multiples() {
        assert_eq!(apply_rotate_to_dims(1000.0, 500.0, 0), (1000.0, 500.0));
        assert_eq!(apply_rotate_to_dims(1000.0, 500.0, 90), (500.0, 1000.0));
        assert_eq!(apply_rotate_to_dims(1000.0, 500.0, 180), (1000.0, 500.0));
        assert_eq!(apply_rotate_to_dims(1000.0, 500.0, 270), (500.0, 1000.0));
        assert_eq!(apply_rotate_to_dims(1000.0, 500.0, 360), (1000.0, 500.0));
        assert_eq!(apply_rotate_to_dims(1000.0, 500.0, -90), (500.0, 1000.0));
    }

    // -----------------------------------------------------------------------
    // strip_matrix — the device-space matrix handed to pdfium's matrix
    // render path. pdfium's own display matrix already applies the page's
    // /Rotate and the y-up -> y-down flip (see the strip_matrix doc comment),
    // so this matrix is a pure scale + downward strip translation, and is
    // completely rotation-independent. These unit tests pin that formula;
    // the end-to-end proof that the rendered pixels match libvips for every
    // /Rotate value lives in libviprs-tests' rotation_libvips_pdfium_parity
    // test, which renders the canonical fixtures through pdfium and compares
    // against committed libvips+pdfium golden images.

    /// Full page (`y_offset = 0`), unit scale: the identity, since pdfium's
    /// own display matrix supplies the rotation, the y-flip, and the fit.
    #[cfg(feature = "pdfium")]
    #[test]
    fn strip_matrix_full_page_is_scale_only() {
        assert_eq!(strip_matrix(1.0, 0), [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    /// `scale` multiplies only the two diagonal terms; the off-diagonal
    /// terms stay zero, so no rotation or shear is ever introduced here.
    #[cfg(feature = "pdfium")]
    #[test]
    fn strip_matrix_scale_only_touches_the_diagonal() {
        assert_eq!(strip_matrix(2.5, 0), [2.5, 0.0, 0.0, 2.5, 0.0, 0.0]);
    }

    /// `y_offset` is a pure downward translation in the `f` term, bringing
    /// display row `y_offset` to the top of the strip bitmap, independent of
    /// scale and of every other coefficient.
    #[cfg(feature = "pdfium")]
    #[test]
    fn strip_matrix_y_offset_is_pure_translation() {
        assert_eq!(strip_matrix(1.0, 50), [1.0, 0.0, 0.0, 1.0, 0.0, -50.0]);
        assert_eq!(strip_matrix(3.0, 120), [3.0, 0.0, 0.0, 3.0, 0.0, -120.0]);
    }

    /// `PageRotation::try_from_degrees` rejects non-multiples of 90 with
    /// the typed [`PdfError::UnsupportedRotation`] variant. The bare
    /// matrix function takes a `PageRotation` and so cannot fail —
    /// invalid rotations are caught at the parsing boundary.
    #[cfg(feature = "pdfium")]
    #[test]
    fn page_rotation_rejects_non_quarter_value() {
        match PageRotation::try_from_degrees(45) {
            Err(PdfError::UnsupportedRotation(45)) => {}
            other => panic!("expected UnsupportedRotation(45), got {other:?}"),
        }
    }

    /// `try_from_degrees` normalises out-of-range rotations via
    /// `rem_euclid 360`. /Rotate -90 is /Rotate 270; /Rotate 450 is
    /// /Rotate 90. Each maps to the canonical [`PageRotation`] variant.
    #[cfg(feature = "pdfium")]
    #[test]
    fn page_rotation_normalises_input() {
        assert_eq!(
            PageRotation::try_from_degrees(-90).unwrap(),
            PageRotation::ThreeQuarter
        );
        assert_eq!(
            PageRotation::try_from_degrees(450).unwrap(),
            PageRotation::Quarter
        );
        assert_eq!(
            PageRotation::try_from_degrees(720).unwrap(),
            PageRotation::Zero
        );
    }

    /// Round-trip: every `PageRotation` value's `as_degrees()` round-
    /// trips through `try_from_degrees`.
    #[cfg(feature = "pdfium")]
    #[test]
    fn page_rotation_degrees_round_trip() {
        for r in [
            PageRotation::Zero,
            PageRotation::Quarter,
            PageRotation::Half,
            PageRotation::ThreeQuarter,
        ] {
            assert_eq!(PageRotation::try_from_degrees(r.as_degrees()).unwrap(), r);
        }
    }

    /// Compress `bytes` with zlib so a decompression bomb can be crafted in a
    /// test: a large run of identical bytes deflates to a tiny stream.
    fn zlib_compress(bytes: &[u8]) -> Vec<u8> {
        use flate2::{Compression, write::ZlibEncoder};
        use std::io::Write;
        let mut enc = ZlibEncoder::new(Vec::new(), Compression::best());
        enc.write_all(bytes).unwrap();
        enc.finish().unwrap()
    }

    /// A few-hundred-byte `/FlateDecode` stream that inflates far past the
    /// per-image cap must yield a typed `DecompressionLimitExceeded` error
    /// rather than allocating gigabytes (the pre-fix `read_to_end` path had no
    /// output bound and OOM-aborted).
    #[test]
    fn flate_decompress_rejects_zip_bomb() {
        // 4 MiB of zeros compresses to a few hundred bytes.
        let bomb = zlib_compress(&vec![0u8; 4 << 20]);
        assert!(
            bomb.len() < FLATE_SLACK_BYTES,
            "compressed bomb should be tiny, got {}",
            bomb.len()
        );
        // Cap derived from a 1x1 declared image (1 * 1 * bpp + slack).
        let cap = MAX_BYTES_PER_PIXEL + FLATE_SLACK_BYTES;
        let err = flate_decompress(&bomb, cap).unwrap_err();
        assert!(
            matches!(err, PdfError::DecompressionLimitExceeded { .. }),
            "expected DecompressionLimitExceeded, got {err:?}"
        );
    }

    /// Output that fits within the cap decompresses normally and round-trips.
    #[test]
    fn flate_decompress_allows_within_cap() {
        let original = b"the quick brown fox".to_vec();
        let compressed = zlib_compress(&original);
        let out = flate_decompress(&compressed, MAX_BYTES_PER_PIXEL + FLATE_SLACK_BYTES).unwrap();
        assert_eq!(out, original);
    }

    /// The cap is derived from the stream's declared dimensions, so a bomb
    /// reached through the real `get_image_data` terminal-`FlateDecode` path is
    /// rejected with a typed error (not an OOM) before any raster is built.
    #[test]
    fn get_image_data_rejects_flate_bomb() {
        use lopdf::{Stream, dictionary};
        let doc = lopdf::Document::with_version("1.5");
        let bomb = zlib_compress(&vec![0u8; 4 << 20]);
        let stream = Stream::new(
            dictionary! {
                "Type" => "XObject",
                "Subtype" => "Image",
                "Width" => 1i64,
                "Height" => 1i64,
                "BitsPerComponent" => 8i64,
                "ColorSpace" => "DeviceRGB",
                "Filter" => "FlateDecode",
            },
            bomb,
        );
        let err = get_image_data(&doc, &stream).unwrap_err();
        assert!(
            matches!(err, PdfError::DecompressionLimitExceeded { .. }),
            "expected DecompressionLimitExceeded, got {err:?}"
        );
    }

    /// The pdfium library must never be resolved from the current working
    /// directory. With no explicit `PDFIUM_PATH` configured, resolution
    /// yields `None` so the caller falls back to the system library search
    /// path — it must not hand back `./` (or any relative cwd marker),
    /// which is the CWE-427 injection vector.
    #[test]
    fn pdfium_path_never_defaults_to_cwd() {
        use std::path::PathBuf;

        // Unset → system fallback (None), never the current directory.
        assert_eq!(resolve_pdfium_path(None), None);
        // Empty value is treated as unset, not as "./".
        assert_eq!(resolve_pdfium_path(Some(std::ffi::OsString::new())), None);

        // Whatever a caller supplies, the resolver must not silently
        // substitute a relative cwd marker.
        for raw in ["/opt/pdfium/lib", "/usr/local/lib/libpdfium.dylib"] {
            let resolved = resolve_pdfium_path(Some(raw.into()))
                .expect("an explicit PDFIUM_PATH resolves to Some");
            assert_eq!(resolved, PathBuf::from(raw));
            assert_ne!(resolved, PathBuf::from("./"));
            assert_ne!(resolved, PathBuf::from("."));
            assert!(
                resolved.is_absolute(),
                "an explicit absolute PDFIUM_PATH stays absolute, not cwd-relative"
            );
        }
    }
}
