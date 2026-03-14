use std::path::Path;

use thiserror::Error;

use crate::pixel::PixelFormat;
use crate::raster::Raster;
use crate::source;

#[derive(Debug, Error)]
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
    #[error("raster error: {0}")]
    Raster(#[from] crate::raster::RasterError),
    #[error("page {page} out of range (document has {total} pages)")]
    PageOutOfRange { page: usize, total: usize },
    #[cfg(feature = "pdfium")]
    #[error("pdfium error: {0}")]
    Pdfium(String),
}

/// Information about a PDF document.
#[derive(Debug, Clone)]
pub struct PdfInfo {
    pub page_count: usize,
    pub pages: Vec<PdfPageInfo>,
}

/// Information about a single PDF page.
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

/// Get information about a PDF document.
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
pub fn extract_page_image(path: &Path, page: usize) -> Result<Raster, PdfError> {
    let doc = lopdf::Document::load(path).map_err(|e| PdfError::Parse(e.to_string()))?;
    let pages_map = doc.get_pages();
    let total = pages_map.len();

    let &page_id = pages_map
        .get(&(page as u32))
        .ok_or(PdfError::PageOutOfRange { page, total })?;

    extract_largest_image(&doc, page_id, page)
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

        // Get image dimensions for size comparison
        let width = stream
            .dict
            .get(b"Width")
            .ok()
            .and_then(|w| w.as_i64().ok())
            .unwrap_or(0) as usize;
        let height = stream
            .dict
            .get(b"Height")
            .ok()
            .and_then(|h| h.as_i64().ok())
            .unwrap_or(0) as usize;
        let pixel_count = width * height;

        if pixel_count == 0 {
            continue;
        }

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
enum ImageData {
    /// Encoded image bytes (JPEG, PNG, JPEG2000) — pass to `decode_bytes`.
    Encoded(Vec<u8>),
    /// Already-decoded raster (from raw FlateDecode pixel data).
    Decoded(Raster),
}

/// Get image data from a PDF stream, handling common filters.
fn get_image_data(doc: &lopdf::Document, stream: &lopdf::Stream) -> Result<ImageData, PdfError> {
    let filter: &[u8] = stream
        .dict
        .get(b"Filter")
        .ok()
        .and_then(|f| f.as_name().ok())
        .unwrap_or(b"");

    match filter {
        b"DCTDecode" => {
            // JPEG data — return raw, let `image` crate decode
            Ok(ImageData::Encoded(stream.content.clone()))
        }
        b"FlateDecode" => {
            // Deflate-compressed raw pixels
            let mut stream_clone = stream.clone();
            stream_clone
                .decompress()
                .map_err(|e| PdfError::Parse(format!("decompress: {e}")))?;
            let decompressed = stream_clone.content;

            let width = stream
                .dict
                .get(b"Width")
                .ok()
                .and_then(|w| w.as_i64().ok())
                .unwrap_or(0) as u32;
            let height = stream
                .dict
                .get(b"Height")
                .ok()
                .and_then(|h| h.as_i64().ok())
                .unwrap_or(0) as u32;
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

            let expected = width as usize * height as usize * format.bytes_per_pixel();
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
        b"JPXDecode" => {
            // JPEG 2000 — return raw, let `image` crate attempt decode
            Ok(ImageData::Encoded(stream.content.clone()))
        }
        b"" => {
            // No filter — raw data, try as image
            Ok(ImageData::Encoded(stream.content.clone()))
        }
        other => Err(PdfError::UnsupportedFormat(
            String::from_utf8_lossy(other).to_string(),
        )),
    }
}

/// Convert CMYK raw bytes to RGB Raster.
fn cmyk_to_rgb_raster(cmyk_data: &[u8], width: u32, height: u32) -> Result<Raster, PdfError> {
    let pixel_count = width as usize * height as usize;
    if cmyk_data.len() < pixel_count * 4 {
        return Err(PdfError::Decode("CMYK data too short".to_string()));
    }
    let mut rgb = Vec::with_capacity(pixel_count * 3);
    for chunk in cmyk_data[..pixel_count * 4].chunks_exact(4) {
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

/// Get page dimensions in points from MediaBox.
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
            return ((x1 - x0).abs(), (y1 - y0).abs());
        }
    }

    (0.0, 0.0)
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

/// Render a PDF page to a raster using PDFium.
///
/// This handles vector content (AutoCAD exports, text, paths) that cannot be
/// extracted as embedded images. Requires the `pdfium` feature and a PDFium
/// library available at runtime.
#[cfg(feature = "pdfium")]
pub fn render_page_pdfium(path: &Path, page: usize, dpi: u32) -> Result<Raster, PdfError> {
    use pdfium_render::prelude::*;

    let pdfium = Pdfium::default();
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

    let scale = dpi as f32 / 72.0;
    let width = (pdf_page.width().value * scale) as u32;
    let height = (pdf_page.height().value * scale) as u32;

    let config = PdfRenderConfig::new()
        .set_target_width(width as i32)
        .set_maximum_height(height as i32);

    let bitmap = pdf_page
        .render_with_config(&config)
        .map_err(|e| PdfError::Pdfium(e.to_string()))?;

    let img = bitmap.as_image();
    let rgba = img.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());
    let data = rgba.into_raw();

    Raster::new(w, h, PixelFormat::Rgba8, data).map_err(PdfError::from)
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
}
