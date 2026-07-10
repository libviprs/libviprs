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
    let pages_map = doc.get_pages();
    let page_count = pages_map.len();
    let mut pages = Vec::with_capacity(page_count);

    // Pages are returned as BTreeMap<u32, ObjectId>, sorted by page number
    for (&page_num, &page_id) in &pages_map {
        let (width_pts, height_pts) = get_page_dimensions(&doc, page_id);
        let has_images = page_has_images(&doc, page_id);

        pages.push(PdfPageInfo {
            page_number: page_num as usize,
            width_pts,
            height_pts,
            has_images,
        });
    }

    Ok(PdfInfo { page_count, pages })
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
    let pages_map = doc.get_pages();
    let total = pages_map.len();

    let &page_id = pages_map
        .get(&(page as u32))
        .ok_or(PdfError::PageOutOfRange { page, total })?;

    extract_largest_image(&doc, page_id, page)
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
    for chunk in cmyk_data[..cmyk_len].chunks_exact(4) {
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
    if let Some(media_box) = resolve_array_entry(doc, dict, b"MediaBox") {
        if media_box.len() >= 4 {
            let x0 = obj_to_f64(&media_box[0]).unwrap_or(0.0);
            let y0 = obj_to_f64(&media_box[1]).unwrap_or(0.0);
            let x1 = obj_to_f64(&media_box[2]).unwrap_or(0.0);
            let y1 = obj_to_f64(&media_box[3]).unwrap_or(0.0);
            let w = (x1 - x0).abs();
            let h = (y1 - y0).abs();
            return apply_rotate_to_dims(w, h, resolve_rotate(doc, page_id));
        }
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
        if let Ok(rotate_obj) = dict.get(b"Rotate") {
            if let Ok(resolved) = resolve_object(doc, rotate_obj) {
                if let Some(v) = obj_to_f64(resolved) {
                    return v as i64;
                }
            }
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
/// This is the path-based companion of the private [`resolve_rotate`]
/// helper. Callers driving pdfium's matrix render path need the page's
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
    let total = pages_map.len();
    let &page_id = pages_map
        .get(&(page as u32))
        .ok_or(PdfError::PageOutOfRange { page, total })?;
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
/// mutex, but **only** when libviprs's `[patch.crates-io]` directive
/// is honoured, which requires the consumer crate (libviprs-tests,
/// libviprs-cli, downstream users) to either share a workspace with
/// libviprs or replicate the patch directive. Cargo does not
/// propagate `[patch.crates-io]` from a path-dependency.
///
/// To keep libviprs correct without depending on consumer-side
/// patch hygiene, every FPDF entry point in this crate acquires this
/// process-wide lock first. If the consumer also has the patched
/// fork active (the recommended setup), the result is double-locking
/// — a few extra nanoseconds per call, no correctness or deadlock
/// concern (the locks are independent `Mutex<()>` instances acquired
/// in a fixed order). If the patch is missing, this lock alone keeps
/// concurrent renders safe.
///
/// **Performance note:** With this lock held, multi-threaded
/// `render_strip` calls serialise. That matches the underlying
/// reality (pdfium itself is single-threaded), so no parallelism is
/// lost — this lock simply makes the serialisation explicit and safe
/// instead of implicit and crashing.
#[cfg(feature = "pdfium")]
static PDFIUM_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Acquire [`PDFIUM_LOCK`]. Panics in another thread that previously
/// held the lock are recovered from rather than propagated; pdfium is
/// a C library and a Rust panic on the Rust side cannot corrupt its
/// internal state, so poisoning here is benign.
#[cfg(feature = "pdfium")]
#[inline]
pub(crate) fn pdfium_lock() -> std::sync::MutexGuard<'static, ()> {
    PDFIUM_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
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
    let _guard = INIT_GUARD.lock().unwrap();
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

/// Convert a rendered pdfium bitmap into an RGBA [`Raster`], defending
/// against the panic-on-mismatch inside `PdfBitmap::as_image()`.
///
/// `as_image` is the bitmap conversion closure (in production,
/// `|| bitmap.as_image()`). The 0.8.x fork's `as_image()` ends in
/// `RgbaImage::from_raw(w, h, bytes).map(...).unwrap()`; `from_raw`
/// returns `None` whenever the width/height pdfium reports disagree with
/// the returned buffer length, so an adversarial or edge page makes
/// `as_image()` panic. Called bare inside a MapReduce render worker,
/// that unwind aborts the rendering thread. Isolating the call in
/// [`std::panic::catch_unwind`] converts the panic into a typed
/// [`PdfError::Pdfium`] the caller can propagate.
///
/// (Upstream pdfium-render 0.9.1's `Result`-returning `as_image`
/// supersedes this guard once the pinned fork is bumped.)
#[cfg(feature = "pdfium")]
fn bitmap_to_raster<F>(as_image: F) -> Result<Raster, PdfError>
where
    F: FnOnce() -> image::DynamicImage,
{
    let img = std::panic::catch_unwind(std::panic::AssertUnwindSafe(as_image)).map_err(|_| {
        PdfError::Pdfium(
            "pdfium reported bitmap dimensions inconsistent with its buffer length \
             (PdfBitmap::as_image mismatch)"
                .to_string(),
        )
    })?;
    let rgba = img.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());
    let data = rgba.into_raw();
    Raster::new(w, h, PixelFormat::Rgba8, data).map_err(PdfError::from)
}

/// Render a page at the given pixel dimensions and return a Raster.
#[cfg(feature = "pdfium")]
pub(crate) fn render_at_size(
    pdf_page: &pdfium_render::prelude::PdfPage<'_>,
    width: u32,
    height: u32,
) -> Result<Raster, PdfError> {
    use pdfium_render::prelude::*;

    let config = PdfRenderConfig::new()
        .set_target_width(width as i32)
        .set_maximum_height(height as i32);

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
    let pdfium = init_pdfium()?;
    let _lock = pdfium_lock();
    let document = pdfium
        .load_pdf_from_file(path, None)
        .map_err(|e| PdfError::Pdfium(e.to_string()))?;

    let pages = document.pages();
    let total = pages.len();
    if page == 0 || page > total as usize {
        return Err(PdfError::PageOutOfRange {
            page,
            total: total as usize,
        });
    }
    let pdf_page = pages
        .get(page as u16 - 1)
        .map_err(|e| PdfError::Pdfium(e.to_string()))?;

    let (width, height) = render_dims_within_budget(
        pdf_page.width().value,
        pdf_page.height().value,
        dpi,
        DEFAULT_MAX_RENDER_PIXELS,
    )?;

    render_at_size(&pdf_page, width, height)
}

/// Compose the device matrix that maps pre-rotation page coordinates
/// (PDF's bottom-left origin, y-up, units = points) into bitmap pixel
/// coordinates (pdfium's top-left origin, y-down, units = pixels) for
/// a strip starting at `y_offset` (in display pixels) of a page rendered
/// at `scale = dpi / 72.0`.
///
/// `display_w_pt` and `display_h_pt` are the **post-/Rotate** page
/// dimensions in points, which is what `pdfium-render`'s
/// `PdfPage::width()/height()` already returns. The function swaps to
/// pre-rotation dimensions internally so callers don't have to.
///
/// The caller supplies the bitmap as `(display_w_px, strip_h_px)` —
/// a strip-sized output canvas — and pdfium fills only the rows
/// `[0, strip_h_px)` because the matrix translates the requested
/// page strip into those rows.
///
/// # Per-rotation matrices
///
/// | `/Rotate` | `[a, b, c, d, e, f]`                                        |
/// |-----------|-------------------------------------------------------------|
/// | 0         | `[s,  0,  0, -s, 0,             s·page_h_pt − y_off]`       |
/// | 90        | `[0,  s,  s,  0, 0,             −y_off]`                    |
/// | 180       | `[−s, 0,  0,  s, s·page_w_pt,   −y_off]`                    |
/// | 270       | `[0, −s, −s,  0, s·page_h_pt,   s·page_w_pt − y_off]`       |
///
/// `s = scale`. `page_w_pt / page_h_pt` are the **pre-rotation** page
/// dimensions (display dims swapped for `/Rotate 90/270`). Derivation:
/// for each rotation, compose page→display map (rotation + y-flip from
/// y-up to y-down) and translate by `-y_offset`.
///
/// Infallible because the typed [`PageRotation`] enum makes invalid
/// rotation values unrepresentable. Errors that previously came from
/// this function (unsupported `/Rotate`) now surface at the parsing
/// boundary in [`PageRotation::try_from_degrees`].
#[cfg(feature = "pdfium")]
#[must_use]
pub(crate) fn strip_matrix(
    rotation: PageRotation,
    scale: f32,
    display_w_pt: f32,
    display_h_pt: f32,
    y_offset: u32,
) -> [f32; 6] {
    let (page_w_pt, page_h_pt) = match rotation {
        PageRotation::Zero | PageRotation::Half => (display_w_pt, display_h_pt),
        PageRotation::Quarter | PageRotation::ThreeQuarter => (display_h_pt, display_w_pt),
    };
    let s = scale;
    let y_off = y_offset as f32;
    match rotation {
        PageRotation::Zero => [s, 0.0, 0.0, -s, 0.0, s * page_h_pt - y_off],
        PageRotation::Quarter => [0.0, s, s, 0.0, 0.0, -y_off],
        PageRotation::Half => [-s, 0.0, 0.0, s, s * page_w_pt, -y_off],
        PageRotation::ThreeQuarter => [0.0, -s, -s, 0.0, s * page_h_pt, s * page_w_pt - y_off],
    }
}

/// Render a single horizontal strip of a PDF page directly via pdfium's
/// matrix render path, allocating only a strip-sized bitmap.
///
/// `y_offset` and `strip_height` are in display-oriented pixel coordinates
/// (top-left origin, y-down) — the same coordinate system the engine and
/// `StripSource` callers already speak. `rotation` is the page's intrinsic
/// `/Rotate` (call [`page_rotate`] to obtain it once per source).
///
/// # Coordinate composition
///
/// `FPDF_RenderPageBitmapWithMatrix` does not auto-apply the page's
/// `/Rotate`; only the form-data render path does. We compose the
/// per-rotation device matrix manually so the rendered strip lands at
/// rows `[0, strip_height)` of the output bitmap. For each `/Rotate`
/// value, the matrix maps pre-rotation page coordinates (y-up, origin
/// bottom-left, units = points) into bitmap pixel coordinates (y-down,
/// origin top-left, units = pixels):
///
/// | `/Rotate` | `[a, b, c, d, e, f]`                                        |
/// |-----------|-------------------------------------------------------------|
/// | 0         | `[s,  0,  0, -s, 0,             s·page_h_pt − y_off]`       |
/// | 90        | `[0,  s,  s,  0, 0,             −y_off]`                    |
/// | 180       | `[−s, 0,  0,  s, s·page_w_pt,   −y_off]`                    |
/// | 270       | `[0, −s, −s,  0, s·page_h_pt,   s·page_w_pt − y_off]`       |
///
/// `s = dpi / 72`. `page_w_pt` / `page_h_pt` are pre-rotation page
/// dimensions in points (swapped from `pdf_page.width()/height()` for
/// `/Rotate 90/270`, since pdfium-render's getters return display-oriented
/// dims — see comment at `pdf.rs:376-379`). The full per-rotation
/// derivation lives in libviprs#70 and is pinned by the rotation cross-
/// product test
/// (`libviprs-tests/tests/pdfium_streaming_rotation_matrix.rs`).
///
/// # Errors
///
/// - [`PdfError::PageOutOfRange`] — `page == 0` or `page > total_pages`.
/// - [`PdfError::Pdfium`] — pdfium load / page get / matrix-validity /
///   render error, including unsupported `/Rotate` values not in
///   `{0, 90, 180, 270}`.
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
/// `sync` feature plus the `[patch.crates-io]` directive in
/// `libviprs/Cargo.toml`).
///
/// `dpi`, `rotation`, `y_offset`, `strip_height` semantics match the
/// matrix derivation documented at [`strip_matrix`].
#[cfg(feature = "pdfium")]
pub(crate) fn render_page_strip_with_page(
    pdf_page: &pdfium_render::prelude::PdfPage<'_>,
    dpi: u32,
    rotation: PageRotation,
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

    let [a, b, c, d, e, f] = strip_matrix(rotation, scale, display_w_pt, display_h_pt, y_offset);
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
    let pdfium = init_pdfium()?;
    let _lock = pdfium_lock();
    let document = pdfium
        .load_pdf_from_file(path, None)
        .map_err(|e| PdfError::Pdfium(e.to_string()))?;

    let pages = document.pages();
    let total = pages.len();
    if page == 0 || page > total as usize {
        return Err(PdfError::PageOutOfRange {
            page,
            total: total as usize,
        });
    }
    let pdf_page = pages
        .get(page as u16 - 1)
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

    /// A dimension/length mismatch from pdfium — the case where
    /// `PdfBitmap::as_image()`'s terminal `RgbaImage::from_raw(w, h,
    /// bytes).map(...).unwrap()` finds `bytes.len() != w * h * 4` and
    /// panics — must surface as a typed [`PdfError::Pdfium`], not unwind
    /// out of the render worker and abort a MapReduce thread. The closure
    /// reproduces that exact terminal expression: a 2x2 RGBA image needs
    /// 16 bytes; it is handed 3, so `from_raw` returns `None` and the
    /// `.unwrap()` panics.
    #[cfg(feature = "pdfium")]
    #[test]
    fn bitmap_to_raster_surfaces_dimension_mismatch_as_error() {
        use image::{DynamicImage, RgbaImage};
        let result = bitmap_to_raster(|| {
            RgbaImage::from_raw(2, 2, vec![0u8; 3])
                .map(DynamicImage::ImageRgba8)
                .unwrap()
        });
        match result {
            Err(PdfError::Pdfium(_)) => {}
            other => panic!("expected Err(PdfError::Pdfium), got {other:?}"),
        }
    }

    /// A well-formed bitmap must still convert into a `Raster` with the
    /// declared dimensions — the panic guard must not reject valid input.
    #[cfg(feature = "pdfium")]
    #[test]
    fn bitmap_to_raster_accepts_valid_image() {
        use image::{DynamicImage, RgbaImage};
        let raster = bitmap_to_raster(|| {
            DynamicImage::ImageRgba8(RgbaImage::from_raw(2, 2, vec![255u8; 16]).unwrap())
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
    // strip_matrix — pure-Rust unit tests for the per-rotation device matrix.
    //
    // These tests verify that the matrix coefficients map page-space corners
    // to the expected bitmap-pixel positions, without going through pdfium.
    // Each rotation is verified independently so a sign error or axis swap
    // surfaces as a single failed test, not a confusing region-mean drift.
    //
    // Apply-the-matrix helper: PDF matrices are [a, b, c, d, e, f] applied
    // as `(x', y') = (a*x + c*y + e, b*x + d*y + f)`. We replicate that
    // here rather than reaching for a 2D linear-algebra crate, since the
    // arithmetic is two multiplies and two adds per coordinate.
    // -----------------------------------------------------------------------
    //
    // Test cases cover: page corners (BL/BR/TR/TL), strip y_offset = 0
    // (full page), strip y_offset > 0 (offset strip), and `scale != 1`.
    // -----------------------------------------------------------------------
    #[cfg(feature = "pdfium")]
    fn apply(matrix: [f32; 6], x: f32, y: f32) -> (f32, f32) {
        let [a, b, c, d, e, f] = matrix;
        (a * x + c * y + e, b * x + d * y + f)
    }

    /// Comparison helper that absorbs f32 rounding (PdfMatrix is f32).
    #[cfg(feature = "pdfium")]
    fn approx_eq(actual: (f32, f32), expected: (f32, f32), label: &str) {
        let dx = (actual.0 - expected.0).abs();
        let dy = (actual.1 - expected.1).abs();
        assert!(
            dx < 1e-3 && dy < 1e-3,
            "{label}: expected ({:.4}, {:.4}), got ({:.4}, {:.4})",
            expected.0,
            expected.1,
            actual.0,
            actual.1
        );
    }

    /// /Rotate 0, scale 1, full-page render (y_offset = 0).
    /// page (0, 0) maps to bitmap (0, page_h) — bottom-left of bitmap.
    /// page (W, H) maps to bitmap (W, 0) — top-right of bitmap.
    #[cfg(feature = "pdfium")]
    #[test]
    fn strip_matrix_rotate_0_full_page_corners() {
        let m = strip_matrix(PageRotation::Zero, 1.0, 100.0, 200.0, 0);
        // Page BL (0, 0) → display BL = bitmap (0, page_h_px) = (0, 200).
        approx_eq(apply(m, 0.0, 0.0), (0.0, 200.0), "/Rotate 0 BL");
        // Page TL (0, H) → display TL = bitmap (0, 0).
        approx_eq(apply(m, 0.0, 200.0), (0.0, 0.0), "/Rotate 0 TL");
        // Page TR (W, H) → display TR = bitmap (W, 0) = (100, 0).
        approx_eq(apply(m, 100.0, 200.0), (100.0, 0.0), "/Rotate 0 TR");
        // Page BR (W, 0) → display BR = bitmap (W, H) = (100, 200).
        approx_eq(apply(m, 100.0, 0.0), (100.0, 200.0), "/Rotate 0 BR");
    }

    /// /Rotate 0, scale 1, strip y_offset = 50.
    /// page TL (0, H) — which would be bitmap (0, 0) at full page — should
    /// move to bitmap (0, -50) (above the strip-sized bitmap).
    /// page (0, H-50) should land at bitmap (0, 0) (the new top of strip).
    #[cfg(feature = "pdfium")]
    #[test]
    fn strip_matrix_rotate_0_strip_offset() {
        let m = strip_matrix(PageRotation::Zero, 1.0, 100.0, 200.0, 50);
        approx_eq(apply(m, 0.0, 200.0), (0.0, -50.0), "/Rotate 0 strip TL");
        approx_eq(apply(m, 0.0, 150.0), (0.0, 0.0), "/Rotate 0 strip y=50 row");
        approx_eq(
            apply(m, 100.0, 150.0),
            (100.0, 0.0),
            "/Rotate 0 strip TR-of-strip",
        );
    }

    /// /Rotate 0, scale 2 — the bitmap is 2x bigger; corners scale linearly.
    #[cfg(feature = "pdfium")]
    #[test]
    fn strip_matrix_rotate_0_scaled() {
        let m = strip_matrix(PageRotation::Zero, 2.0, 100.0, 200.0, 0);
        approx_eq(apply(m, 0.0, 0.0), (0.0, 400.0), "/Rotate 0 scale=2 BL");
        approx_eq(apply(m, 100.0, 200.0), (200.0, 0.0), "/Rotate 0 scale=2 TR");
    }

    /// /Rotate 90, scale 1, full page. Display orientation: landscape
    /// (display_w_pt = page_h_pt = 200, display_h_pt = page_w_pt = 100).
    /// Page BL (0, 0) → display top-left = bitmap (0, 0).
    /// Page TR (page_w_pt, page_h_pt) = (100, 200) → display bottom-right.
    #[cfg(feature = "pdfium")]
    #[test]
    fn strip_matrix_rotate_90_full_page_corners() {
        // Caller passes display dims; for /Rotate 90 portrait original
        // (page_w_pt=100, page_h_pt=200), display is landscape:
        // display_w_pt = page_h_pt = 200, display_h_pt = page_w_pt = 100.
        let m = strip_matrix(PageRotation::Quarter, 1.0, 200.0, 100.0, 0);
        // Page BL (0, 0) → display top-left = bitmap (0, 0).
        approx_eq(
            apply(m, 0.0, 0.0),
            (0.0, 0.0),
            "/Rotate 90 page-BL → display-TL",
        );
        // Page TL (0, page_h_pt=200) → display top-right = bitmap (200, 0).
        approx_eq(
            apply(m, 0.0, 200.0),
            (200.0, 0.0),
            "/Rotate 90 page-TL → display-TR",
        );
        // Page TR (page_w=100, page_h=200) → display bottom-right = (200, 100).
        approx_eq(
            apply(m, 100.0, 200.0),
            (200.0, 100.0),
            "/Rotate 90 page-TR → display-BR",
        );
        // Page BR (100, 0) → display bottom-left = (0, 100).
        approx_eq(
            apply(m, 100.0, 0.0),
            (0.0, 100.0),
            "/Rotate 90 page-BR → display-BL",
        );
    }

    /// /Rotate 180, scale 1, full page. Display orientation: same as
    /// page orientation (rotation is a half-turn). Page BL → display
    /// top-right; page TR → display bottom-left.
    #[cfg(feature = "pdfium")]
    #[test]
    fn strip_matrix_rotate_180_full_page_corners() {
        // Display dims = page dims for /Rotate 180.
        let m = strip_matrix(PageRotation::Half, 1.0, 100.0, 200.0, 0);
        // Page BL (0, 0) → display top-right = bitmap (W=100, 0).
        approx_eq(
            apply(m, 0.0, 0.0),
            (100.0, 0.0),
            "/Rotate 180 page-BL → display-TR",
        );
        // Page TR (100, 200) → display bottom-left = bitmap (0, H=200).
        approx_eq(
            apply(m, 100.0, 200.0),
            (0.0, 200.0),
            "/Rotate 180 page-TR → display-BL",
        );
        // Page BR (100, 0) → display top-left = bitmap (0, 0).
        approx_eq(
            apply(m, 100.0, 0.0),
            (0.0, 0.0),
            "/Rotate 180 page-BR → display-TL",
        );
        // Page TL (0, 200) → display bottom-right = bitmap (100, 200).
        approx_eq(
            apply(m, 0.0, 200.0),
            (100.0, 200.0),
            "/Rotate 180 page-TL → display-BR",
        );
    }

    /// /Rotate 270, scale 1, full page. Display orientation: landscape
    /// (display_w_pt = page_h_pt, display_h_pt = page_w_pt).
    /// Page BL (0, 0) → display bottom-right.
    /// Page TR → display top-left.
    #[cfg(feature = "pdfium")]
    #[test]
    fn strip_matrix_rotate_270_full_page_corners() {
        // Caller passes display dims; for /Rotate 270 portrait (W=100, H=200),
        // display is landscape: display_w_pt=200, display_h_pt=100.
        let m = strip_matrix(PageRotation::ThreeQuarter, 1.0, 200.0, 100.0, 0);
        // Page BL (0, 0) → display bottom-right = bitmap (display_w_pt=200, display_h_pt=100).
        approx_eq(
            apply(m, 0.0, 0.0),
            (200.0, 100.0),
            "/Rotate 270 page-BL → display-BR",
        );
        // Page TR (page_w=100, page_h=200) → display top-left = (0, 0).
        approx_eq(
            apply(m, 100.0, 200.0),
            (0.0, 0.0),
            "/Rotate 270 page-TR → display-TL",
        );
        // Page BR (100, 0) → display top-right = (200, 0).
        approx_eq(
            apply(m, 100.0, 0.0),
            (200.0, 0.0),
            "/Rotate 270 page-BR → display-TR",
        );
        // Page TL (0, 200) → display bottom-left = (0, 100).
        approx_eq(
            apply(m, 0.0, 200.0),
            (0.0, 100.0),
            "/Rotate 270 page-TL → display-BL",
        );
    }

    /// Strip y_offset translates the result by -y_offset for every rotation,
    /// without changing the rotation algebra. This invariant is what lets
    /// the engine's "render strip K, then strip K+1" loop work.
    #[cfg(feature = "pdfium")]
    #[test]
    fn strip_matrix_y_offset_pure_translation_all_rotations() {
        // For each rotation, computing `apply(matrix(rot, ..., y_off=0), x, y)`
        // and `apply(matrix(rot, ..., y_off=K), x, y)` must produce results
        // that differ only by `(0, -K)` in bitmap coords.
        for &rot in &[
            PageRotation::Zero,
            PageRotation::Quarter,
            PageRotation::Half,
            PageRotation::ThreeQuarter,
        ] {
            // Use display dims that fit each rotation. For /Rotate 0/180,
            // display = (W=100, H=200). For 90/270, display = (W=200, H=100).
            let (dw, dh): (f32, f32) = match rot {
                PageRotation::Zero | PageRotation::Half => (100.0, 200.0),
                PageRotation::Quarter | PageRotation::ThreeQuarter => (200.0, 100.0),
            };
            let m_no_off = strip_matrix(rot, 1.0, dw, dh, 0);
            let m_with_off = strip_matrix(rot, 1.0, dw, dh, 50);
            // Pick an arbitrary page point inside any of the four rotation's
            // bounding boxes (use the smaller of dimensions).
            let (test_x, test_y) = (10.0_f32, 10.0_f32);
            let p0 = apply(m_no_off, test_x, test_y);
            let p1 = apply(m_with_off, test_x, test_y);
            approx_eq(
                (p1.0 - p0.0, p1.1 - p0.1),
                (0.0, -50.0),
                &format!("/Rotate {rot:?} y_offset translation invariant"),
            );
        }
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
