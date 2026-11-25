//! Drawing an OLE2FRAME from the picture it carries (issue #1705).
//!
//! The provider hands over the frame's corners and its cached preview ([`OleFrame`]). An enhanced
//! metafile is played by [`emf::play`] into lines, solid cells and text, all in the frame's own
//! coordinates, and a raster or a DIB is placed like an IMAGE (#1667): behind the same pixel limits,
//! with a frame made from the corners. Nothing is read from disk, and a linked object draws its
//! cached picture only. A Windows metafile is not played, and says so.
//!
//! Every outcome is a note for the entity's row, and a refusal is a note too: a hostile preview
//! costs the frame its picture and never the sheet.

use ::image::{DynamicImage, RgbaImage};

use super::dwg_image;
use crate::pvec::emf::{self, EmfShape};
use crate::pvec::{Annotation, ImageFrame, OleFrame, OlePreview, Point3, RawVertex, TextAnchorV};
use crate::raster_pvec::pyramid::RasterPyramid;

/// What a frame's preview comes to.
pub(super) struct OleDrawn {
    /// The polylines to draw, in the frame's block coordinates, and their height.
    pub lines: Vec<Vec<[f64; 2]>>,
    /// The height the shapes sit at.
    pub elevation: f64,
    /// The solid cells and the text, to carry in the streams.
    pub runs: Vec<Annotation>,
    /// A raster or DIB preview: its frame, its bytes and whether the bytes are a DIB.
    pub picture: Option<(ImageFrame, Vec<u8>, bool)>,
    /// The lines of the entity's row.
    pub notes: Vec<String>,
}

/// The shapes of `frame`'s preview, or why it draws nothing.
pub(super) fn draw(frame: OleFrame) -> OleDrawn {
    let (ul, lr) = (frame.upper_left, frame.lower_right);
    let mut out = OleDrawn {
        lines: Vec::new(),
        elevation: ul[2],
        runs: Vec::new(),
        picture: None,
        notes: Vec::new(),
    };
    let refuse =
        |out: &mut OleDrawn, why: &str| out.notes.push(format!("OLE2FRAME: not drawn, {why}"));
    // The most bytes of a raster or DIB preview; the preview's bytes move into the result, so a
    // frame never holds two copies of them.
    const MAX_PICTURE_BYTES: usize = 32 << 20;
    let mut picture = |out: &mut OleDrawn, bytes: Vec<u8>, dib: bool| {
        if bytes.len() > MAX_PICTURE_BYTES {
            out.notes.push(format!(
                "OLE2FRAME: not drawn, its preview is over {MAX_PICTURE_BYTES} bytes"
            ));
            return;
        }
        // One pixel is the whole frame, so the picture's own size never matters to its place:
        // pixel (s, t) of the unit square is at the lower-left corner plus s along the frame's
        // width and t up its height.
        let at = |i: usize| (ul[i].min(lr[i]), (ul[i] - lr[i]).abs());
        let ((x, w), (y, h)) = (at(0), at(1));
        let image = ImageFrame {
            file_name: String::new(),
            insert: [x, y, ul[2]],
            u: [w, 0.0, 0.0],
            v: [0.0, h, 0.0],
            size: [1.0, 1.0],
            visible: true,
            clip: Vec::new(),
            clip_inverted: false,
        };
        out.picture = Some((image, bytes, dib));
    };
    match frame.preview {
        None => refuse(&mut out, "it carries no preview"),
        Some(OlePreview::Wmf(_)) => refuse(&mut out, "a Windows metafile preview is not played"),
        Some(OlePreview::Emf(bytes)) => match emf::play(&bytes, [ul[0], ul[1]], [lr[0], lr[1]]) {
            Err(why) => refuse(&mut out, &why.to_string()),
            Ok(drawing) => {
                out.notes
                    .push("OLE2FRAME: drawn from its EMF preview".to_owned());
                out.notes
                    .extend(drawing.notes.into_iter().map(|n| format!("OLE2FRAME: {n}")));
                let normal = [0.0, 0.0, 1.0];
                for shape in drawing.shapes {
                    match shape {
                        EmfShape::Line(points) => out.lines.push(points),
                        EmfShape::Fill { ring, rgba } => out.runs.push(Annotation::Fill {
                            rings: vec![
                                ring.iter()
                                    .map(|p| RawVertex {
                                        x: p[0],
                                        y: p[1],
                                        bulge: 0.0,
                                    })
                                    .collect(),
                            ],
                            elevation: ul[2],
                            normal,
                            pattern: None,
                            rgba,
                            unread_loops: 0,
                            pattern_lines: Vec::new(),
                        }),
                        EmfShape::Text {
                            at,
                            height,
                            rotation,
                            anchor,
                            text,
                        } => {
                            out.runs.push(Annotation::Text {
                                position: [at[0], at[1], ul[2]],
                                height,
                                rotation,
                                content: text,
                                normal,
                                width_factor: 1.0,
                                anchor: (anchor, TextAnchorV::Baseline),
                                fit: None,
                                block_width: 0.0,
                                line_spacing: 1.0,
                            });
                        }
                    }
                }
            }
        },
        Some(OlePreview::Raster(bytes)) => picture(&mut out, bytes, false),
        Some(OlePreview::Dib(bytes)) => picture(&mut out, bytes, true),
    }
    out
}

/// The size in pixels a raster or DIB preview declares, from its header alone.
pub(super) fn pixels(bytes: &[u8], dib: bool) -> Result<u64, String> {
    let (w, h) = match (dib, bytes.strip_prefix(b"BM")) {
        (true, _) => dib_size(bytes)?,
        (false, Some(rest)) if rest.len() > 12 => dib_size(&bytes[14..])?,
        _ => ::image::ImageReader::new(std::io::Cursor::new(bytes))
            .with_guessed_format()
            .map_err(|e| e.to_string())?
            .into_decoder()
            .map(|d| ::image::ImageDecoder::dimensions(&d))
            .map_err(|e| format!("its preview does not decode ({e})"))?,
    };
    Ok(u64::from(w) * u64::from(h))
}

/// The width and height a DIB header declares.
fn dib_size(b: &[u8]) -> Result<(u32, u32), String> {
    let at = |i: usize| {
        b.get(i..i + 4)
            .map(|s| i32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    };
    match (at(4), at(8)) {
        (Some(w), Some(h)) if w > 0 && h != 0 => Ok((w as u32, h.unsigned_abs())),
        _ => Err("its DIB is cut short".to_owned()),
    }
}

/// The pyramid of a raster preview placed by `frame`, as [`dwg_image::place`] does for a file.
pub(super) fn place(
    frame: &ImageFrame,
    bytes: &[u8],
    dib: bool,
    map: impl Fn(Point3) -> [f64; 2],
    clip_box: Option<[f64; 4]>,
    ring: Option<&[[f64; 2]]>,
    max_pixels: u64,
) -> Result<RasterPyramid, String> {
    let picture = decode(bytes, dib, max_pixels.min(dwg_image::MAX_PIXELS))?;
    dwg_image::place_picture(frame, picture, map, clip_box, ring)
}

/// The picture of a raster preview, decoded within `max_pixels`.
fn decode(bytes: &[u8], dib: bool, max_pixels: u64) -> Result<DynamicImage, String> {
    match (dib, bytes.strip_prefix(b"BM")) {
        (true, _) => decode_dib(bytes, max_pixels),
        // A BMP file: its 14 byte header, then a DIB.
        (false, Some(rest)) if rest.len() > 12 => decode_dib(&bytes[14..], max_pixels),
        _ => dwg_image::decode(bytes.to_vec(), "its preview", max_pixels),
    }
}

/// A DIB (`BITMAPINFOHEADER` or later, then the bits) of 8, 24 or 32 bits a pixel without
/// compression. The header's size is checked before a pixel is allocated.
fn decode_dib(b: &[u8], max_pixels: u64) -> Result<DynamicImage, String> {
    let u32_at = |at: usize| {
        b.get(at..at + 4)
            .map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
    };
    let (Some(header), Some(w), Some(h), Some(bits), Some(compression), Some(used)) = (
        u32_at(0),
        u32_at(4).map(|v| v as i32),
        u32_at(8).map(|v| v as i32),
        b.get(14..16).map(|s| u16::from_le_bytes([s[0], s[1]])),
        u32_at(16),
        u32_at(32),
    ) else {
        return Err("its DIB is cut short".to_owned());
    };
    if header < 40 || compression != 0 || !matches!(bits, 8 | 24 | 32) {
        return Err("its DIB is not an uncompressed 8, 24 or 32 bit picture".to_owned());
    }
    let (w, rows) = (i64::from(w), i64::from(h).abs());
    if w <= 0 || rows == 0 || w > 1 << 16 || rows > 1 << 16 || (w * rows) as u64 > max_pixels {
        return Err(format!(
            "its DIB declares {w} x {rows} pixels, over the {max_pixels} it may have"
        ));
    }
    let palette = if bits == 8 {
        if used == 0 {
            256
        } else {
            used.min(256) as usize
        }
    } else {
        0
    };
    let stride = ((w as usize * usize::from(bits)).div_ceil(32)) * 4;
    let start = header as usize + palette * 4;
    let bits_end = start.checked_add(stride * rows as usize);
    if bits_end.is_none_or(|end| end > b.len()) {
        return Err("its DIB is cut short".to_owned());
    }
    let mut image = RgbaImage::new(w as u32, rows as u32);
    for y in 0..rows as usize {
        // A positive height is stored bottom row first.
        let source = if h > 0 { rows as usize - 1 - y } else { y };
        let row = &b[start + source * stride..][..stride];
        for x in 0..w as usize {
            let rgb = match bits {
                8 => {
                    let i = usize::from(row[x]).min(palette.max(1) - 1) * 4 + header as usize;
                    [b[i + 2], b[i + 1], b[i]]
                }
                24 => [row[x * 3 + 2], row[x * 3 + 1], row[x * 3]],
                _ => [row[x * 4 + 2], row[x * 4 + 1], row[x * 4]],
            };
            image.put_pixel(
                x as u32,
                y as u32,
                ::image::Rgba([rgb[0], rgb[1], rgb[2], 255]),
            );
        }
    }
    Ok(DynamicImage::ImageRgba8(image))
}
