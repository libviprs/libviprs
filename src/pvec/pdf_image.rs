//! PDF image placements: decode, bake and cut the images of a page into PVec rasters
//! (issue #1332, RL2.1 of EPIC R-L, ADR 0018 in libviprs/pvec).
//!
//! A PDF with a scanned underlay or an aerial converts to linework and nothing else unless the
//! images come along. [`bake_page_images`] takes the images [`PdfVectorPage::images`] recorded
//! and, for each one, decodes the stream, bakes the rules ADR 0018 puts on a PDF producer, and
//! cuts the pyramid a PVec raster stream stores ([`RasterPyramid`]).
//!
//! # The bake rules
//!
//! * Colour: `DeviceGray`, `DeviceRGB` and `DeviceCMYK` at 1, 2, 4, 8 and 16 bits (16 is reduced
//!   to 8 and reported), `/Indexed` expanded through its lookup, `/ICCBased` read as the device
//!   space of its `/N`, `/CalGray`, `/CalRGB` and `/Lab` into sRGB by the standard formulas. Every
//!   conversion that is not exact (ICC, Cal, Lab, CMYK) is reported as
//!   `colour-space-approximated`.
//! * `/Decode` is applied to every component. `/ImageMask true` is painted in the fill colour
//!   in force at the `Do` or `BI`, into RGBA, with the unpainted pixels transparent in that same
//!   colour. `/SMask` is resampled to the image size by nearest pixel centre and written into
//!   alpha, and an explicit `/Mask` is ignored and reported.
//! * A 1-bit gray image with no alpha is cut into 1-bit PNG tiles (`bilevel-1-bit-tiles`).
//! * Streams: no filter, `FlateDecode` (with the TIFF and PNG predictors), `DCTDecode` (decoded
//!   and re-tiled, because the pyramid is a different grid; reported as a re-encode of a lossy
//!   source), Flate over DCT, and `CCITTFaxDecode` (through `hayro-ccitt`, in every build, issue
//!   #1363). `JPXDecode` and `JBIG2Decode` are refused as `filter-not-enabled:<name>` (this
//!   build has no codec for them yet), and any other filter as
//!   `unsupported-filter:<name>`. A codec that was tried and refused the stream is `decode-failed`.
//!   [`PdfImageReport::written_by_filter`] counts the written images by their last filter.
//! * Refused by name too: `/Separation`, `/DeviceN` and `/Pattern`, an image under a
//!   graphics-state soft mask, and an image whose CTM is singular or not finite.
//! * Placement: the unit square through the CTM, `x = A*u + C*v + E` with `(u, v) = (px/w,
//!   1 - py/h)`, which is `[A/w -C/h B/w -D/h C+E D+F]` in PVec's `Transform2` order, composed
//!   with [`PdfImageOptions::page_to_source`] and checked against NP-19 before the write.
//! * Clips: every clip in force is intersected. A rectangle that is axis-aligned in pixel space
//!   and lands on pixel edges only sets `visible_box`. Any other clip is rasterised at image
//!   resolution into alpha (pixel centres, the clip's own winding rule) and `visible_box` is the
//!   clips' bounding box inside the image. A baked alpha channel needs PNG or WebP tiles
//!   (RASTER-04), and JPEG tiles are refused for it here.
//! * Order: `RasterId` is the image's position among the written ones, in content
//!   order. An image painted before every painted path is the underlay band, after every one the
//!   overlay band, and between paths it is the underlay with `interleaved-draw-order` reported,
//!   which is the v1 limit the ADR records.
//!
//! Every image is accounted for: [`PdfImageReport::images`] equals
//! [`PdfImageReport::written`] plus [`PdfImageReport::refused_total`].

use std::collections::BTreeMap;

use thiserror::Error;

use super::pdf_clip::exact_rect;
use super::pdf_image_decode::{Channels, Decoded, decode};
use super::pdf_vector::{PdfImagePlacement, PdfPathEntity, PdfPathSegment, PdfVectorPage};
use crate::jp2k::Jp2kError;
use crate::pdf::PdfError;
use crate::pixel::PixelFormat;
use crate::planner::{Layout, PyramidPlanner};
use crate::raster::{Raster, RasterError};
use crate::raster_pvec::pyramid::{
    RasterBand, RasterPlacement, RasterPyramid, RasterPyramidError, RasterPyramidOptions,
    produce_raster_pyramid,
};
use crate::sink::TileFormat;
use crate::source::SourceError;

/// What a caller chooses about the bake.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct PdfImageOptions {
    /// How tiles are encoded. A baked alpha channel needs PNG or WebP (RASTER-04).
    pub tile_format: TileFormat,
    /// 256, 512 or 1024.
    pub tile_edge: u32,
    /// The map from PDF page space into PVec source space, `[a b c d tx ty]` in PVec's order.
    /// Page space is already y-up with the displayed page's lower-left corner as the origin, so
    /// the identity is the default.
    pub page_to_source: [f64; 6],
    /// The most pixels one baked image may have.
    pub max_pixels: u64,
}

impl Default for PdfImageOptions {
    fn default() -> Self {
        Self {
            tile_format: TileFormat::Png,
            tile_edge: 256,
            page_to_source: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            max_pixels: 64 << 20,
        }
    }
}

/// A thing the bake did that the reader of the report should know.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[non_exhaustive]
pub enum PdfImageNote {
    /// An ICC, Cal or Lab colour space (or CMYK) was converted to sRGB by a standard formula.
    ColourSpaceApproximated,
    /// 16 bits per component were reduced to 8.
    BitDepthReduced,
    /// A `DCTDecode` image was decoded and re-tiled: the tiles are a re-encode of a lossy source.
    ReEncodedLossy,
    /// The image was painted between paths, and the band cannot say so.
    Interleaved,
    /// A clip was baked into alpha or into `visible_box`.
    Clipped,
    /// An explicit `/Mask` (colour key or stencil) was not applied.
    MaskIgnored,
    /// A 1-bit gray image was cut into 1-bit tiles.
    Bilevel,
}

impl PdfImageNote {
    /// The report line.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ColourSpaceApproximated => "colour-space-approximated",
            Self::BitDepthReduced => "bit-depth-reduced",
            Self::ReEncodedLossy => "re-encoded-lossy",
            Self::Interleaved => "interleaved-draw-order",
            Self::Clipped => "clipped",
            Self::MaskIgnored => "explicit-mask-ignored",
            Self::Bilevel => "bilevel-1-bit-tiles",
        }
    }
}

/// Why an image was not written.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PdfImageError {
    /// The image is not well formed: a missing entry, a stream shorter than its size, a
    /// parameter outside its range.
    #[error("image decode failed: {0}")]
    Decode(String),
    /// More pixels than [`PdfImageOptions::max_pixels`].
    #[error("{pixels} pixels is more than the limit of {limit}")]
    TooLarge {
        /// The image's `width * height`.
        pixels: u64,
        /// The limit.
        limit: u64,
    },
    /// A stream could not be inflated.
    #[error("the image stream could not be inflated")]
    Pdf(#[source] PdfError),
    /// The baked samples do not make a raster.
    #[error("the baked samples are not a raster")]
    Raster(#[source] RasterError),
    /// A filter nobody decodes (`LZWDecode`, `ASCII85Decode`...).
    #[error("unsupported filter {0}")]
    UnsupportedFilter(String),
    /// A filter this build was compiled without. [`reason`](Self::reason) is
    /// `filter-not-enabled:<filter>`.
    #[error("filter {filter} needs the `{feature}` feature")]
    FilterNotEnabled {
        /// The filter name as the PDF spells it.
        filter: String,
        /// The Cargo feature that would decode it.
        feature: &'static str,
    },
    /// A codec refused the stream. [`reason`](Self::reason) stays `decode-failed`.
    #[error("the {filter} stream did not decode")]
    Filter {
        /// The filter name as the PDF spells it.
        filter: String,
        /// What the codec said.
        #[source]
        source: FilterError,
    },
    /// One image stream is over [`super::pdf_vector::PdfVectorLimits::max_image_stream_bytes`].
    #[error("the image stream is {bytes} bytes, the per-stream limit is {limit}")]
    StreamTooLarge {
        /// The encoded size.
        bytes: u64,
        /// The limit.
        limit: u64,
    },
    /// A colour space the bake does not take (`Separation`, `DeviceN`, `Pattern`).
    #[error("unsupported colour space {0}")]
    UnsupportedColourSpace(String),
    /// A graphics-state soft mask was in force.
    #[error("a graphics-state soft mask is in force")]
    GraphicsStateSoftMask,
    /// The CTM is singular or not finite.
    #[error("the image placement is singular or not finite")]
    SingularCtm,
    /// The placement fails NP-19.
    #[error("the image placement fails NP-19: {0}")]
    InvalidPlacement(&'static str),
    /// The clips leave nothing of the image.
    #[error("the clips leave nothing of the image")]
    FullyClipped,
    /// A baked alpha channel with JPEG tiles (RASTER-04).
    #[error("RASTER-04: a baked alpha channel cannot be stored in JPEG tiles")]
    AlphaNeedsLosslessTiles,
    /// The cut pyramid was refused.
    #[error("the pyramid was refused")]
    Pyramid(#[source] RasterPyramidError),
}

/// What a filter codec said when it refused a stream.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum FilterError {
    /// The JPEG 2000 decoder refused it, or the build has none.
    #[error(transparent)]
    Jp2k(Jp2kError),
    /// The JPEG decoder refused it.
    #[error(transparent)]
    Dct(SourceError),
    /// A shared decode ceiling, or a refusal that is not the codec's own (`JPXDecode`: width,
    /// height or allocation over the budget).
    #[error(transparent)]
    Source(SourceError),
    /// `CCITTFaxDecode`: the codec's message.
    #[error("CCITT: {0}")]
    Ccitt(String),
    /// `JBIG2Decode`: the codec's message.
    #[error("JBIG2: {0}")]
    Jbig2(String),
    /// The stream decoded to fewer bytes than the image needs.
    #[error("the stream decoded to {got} bytes, the image needs {need}")]
    ShortOutput {
        /// Bytes the codec produced.
        got: usize,
        /// Bytes the dictionary promised.
        need: usize,
    },
    /// `/SMaskInData` asked for something that is not applied.
    #[error("/SMaskInData {0} is not applied")]
    SMaskInData(i64),
    /// A `/DecodeParms` entry that disagrees with the dictionary or is out of range.
    #[error("/DecodeParms: {0}")]
    Parms(String),
}

impl PdfImageError {
    /// The stable reason the report counts it under.
    #[must_use]
    pub fn reason(&self) -> String {
        match self {
            Self::Decode(_) | Self::Pdf(_) | Self::Raster(_) | Self::Filter { .. } => {
                "decode-failed".to_string()
            }
            Self::TooLarge { .. } | Self::StreamTooLarge { .. } => "too-large".to_string(),
            Self::UnsupportedFilter(n) => format!("unsupported-filter:{n}"),
            Self::FilterNotEnabled { filter, .. } => format!("filter-not-enabled:{filter}"),
            Self::UnsupportedColourSpace(n) => format!("unsupported-colour-space:{n}"),
            Self::GraphicsStateSoftMask => "gs-soft-mask".to_string(),
            Self::SingularCtm => "singular-ctm".to_string(),
            Self::InvalidPlacement(_) => "invalid-placement".to_string(),
            Self::FullyClipped => "fully-clipped".to_string(),
            Self::AlphaNeedsLosslessTiles => "raster-04-alpha-needs-lossless".to_string(),
            Self::Pyramid(_) => "pyramid-refused".to_string(),
        }
    }
}

/// One image that was written.
#[derive(Debug)]
#[non_exhaustive]
pub struct PdfBakedImage {
    /// The index in [`PdfVectorPage::images`].
    pub index: usize,
    /// The cut pyramid. The `RasterId` is this image's position in [`PdfImageBake::written`].
    pub pyramid: RasterPyramid,
    /// What the bake did.
    pub notes: Vec<PdfImageNote>,
}

/// One image that was not written.
#[derive(Debug)]
#[non_exhaustive]
pub struct PdfRefusedImage {
    /// The index in [`PdfVectorPage::images`].
    pub index: usize,
    /// Why.
    pub error: PdfImageError,
}

/// Counts per outcome, beside the fill constructs.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct PdfImageReport {
    /// Images on the page (image XObjects and inline images).
    pub images: u64,
    /// Images written.
    pub written: u64,
    /// Images refused, by reason.
    pub refused: BTreeMap<String, u64>,
    /// Written images whose colour was approximated.
    pub colour_approximated: u64,
    /// Written images reduced from 16 bits.
    pub bit_depth_reduced: u64,
    /// Written images that are a re-encode of a lossy source.
    pub re_encoded_lossy: u64,
    /// Written images painted between paths.
    pub interleaved: u64,
    /// Written images with a clip baked.
    pub clipped: u64,
    /// Written images whose explicit `/Mask` was ignored.
    pub mask_ignored: u64,
    /// Written images cut into 1-bit tiles.
    pub bilevel: u64,
    /// Written images by the filter that produced their samples: the last filter of the chain,
    /// or `none` for an unfiltered image (issue #1362).
    pub written_by_filter: BTreeMap<String, u64>,
}

impl PdfImageReport {
    /// Images refused, all reasons.
    #[must_use]
    pub fn refused_total(&self) -> u64 {
        self.refused.values().sum()
    }
}

/// The result of baking a page's images.
#[derive(Debug)]
#[non_exhaustive]
pub struct PdfImageBake {
    /// The written images in content-stream order. A `RasterId` is a position here.
    pub written: Vec<PdfBakedImage>,
    /// The refused images in content-stream order.
    pub refused: Vec<PdfRefusedImage>,
    /// The counts.
    pub report: PdfImageReport,
}

/// Bake every image of `page`, in content-stream order.
#[must_use]
pub fn bake_page_images(page: &PdfVectorPage, options: &PdfImageOptions) -> PdfImageBake {
    let painted: Vec<u64> = page
        .entities
        .iter()
        .filter(|e| e.paint.fill.is_some() || e.paint.stroke)
        .map(|e| e.provenance.sequence)
        .collect();
    let mut bake = PdfImageBake {
        written: Vec::new(),
        refused: Vec::new(),
        report: PdfImageReport {
            images: page.images.len() as u64,
            ..PdfImageReport::default()
        },
    };
    for (index, image) in page.images.iter().enumerate() {
        match bake_one(page, image, &painted, options) {
            Ok((pyramid, notes, filter)) => {
                let r = &mut bake.report;
                r.written += 1;
                for note in &notes {
                    let counter = match note {
                        PdfImageNote::ColourSpaceApproximated => &mut r.colour_approximated,
                        PdfImageNote::BitDepthReduced => &mut r.bit_depth_reduced,
                        PdfImageNote::ReEncodedLossy => &mut r.re_encoded_lossy,
                        PdfImageNote::Interleaved => &mut r.interleaved,
                        PdfImageNote::Clipped => &mut r.clipped,
                        PdfImageNote::MaskIgnored => &mut r.mask_ignored,
                        PdfImageNote::Bilevel => &mut r.bilevel,
                    };
                    *counter += 1;
                }
                *r.written_by_filter.entry(filter).or_insert(0) += 1;
                bake.written.push(PdfBakedImage {
                    index,
                    pyramid,
                    notes,
                });
            }
            Err(error) => {
                *bake.report.refused.entry(error.reason()).or_insert(0) += 1;
                bake.refused.push(PdfRefusedImage { index, error });
            }
        }
    }
    bake
}

fn bake_one(
    page: &PdfVectorPage,
    image: &PdfImagePlacement,
    painted: &[u64],
    options: &PdfImageOptions,
) -> Result<(RasterPyramid, Vec<PdfImageNote>, String), PdfImageError> {
    if image.soft_masked {
        return Err(PdfImageError::GraphicsStateSoftMask);
    }
    let [a, b, c, d, e, f] = image.ctm;
    let det = a * d - b * c;
    if !image.ctm.iter().all(|v| v.is_finite()) || det == 0.0 || !det.is_finite() {
        return Err(PdfImageError::SingularCtm);
    }
    let (Some(w), Some(h)) = (image.width, image.height) else {
        return Err(PdfImageError::Decode(
            "missing or invalid /Width or /Height".to_string(),
        ));
    };
    // Pixel (px, py) is (u, v) = (px / w, 1 - py / h) in the unit square.
    let (wf, hf) = (f64::from(w), f64::from(h));
    let to_page = [a / wf, -c / hf, b / wf, -d / hf, c + e, d + f];
    let pixel_to_source = placement(to_page, options.page_to_source)?;

    let clips = clip_plan(page, image, to_page, w, h)?;
    if !clips.polygons.is_empty() && matches!(options.tile_format, TileFormat::Jpeg { .. }) {
        return Err(PdfImageError::AlphaNeedsLosslessTiles);
    }

    let decoded = decode(image, options.max_pixels)?;
    let mut notes = decoded.notes.clone();
    let mut decoded = decoded;
    if !clips.polygons.is_empty() {
        decoded = apply_alpha(decoded, &clips);
        if decoded.data.chunks(4).all(|p| p[3] == 0) {
            return Err(PdfImageError::FullyClipped);
        }
    }
    if decoded.channels == Channels::Rgba && matches!(options.tile_format, TileFormat::Jpeg { .. })
    {
        return Err(PdfImageError::AlphaNeedsLosslessTiles);
    }
    if clips.visible_box != [0, 0, w, h] || !clips.polygons.is_empty() {
        notes.push(PdfImageNote::Clipped);
    }

    let format = match decoded.channels {
        Channels::Gray => PixelFormat::Gray8,
        Channels::Rgb => PixelFormat::Rgb8,
        Channels::Rgba => PixelFormat::Rgba8,
    };
    let raster = Raster::new(w, h, format, decoded.data).map_err(PdfImageError::Raster)?;
    let plan = PyramidPlanner::new(w, h, options.tile_edge, 0, Layout::Google)
        .map_err(|e| PdfImageError::Pyramid(RasterPyramidError::Plan(e)))?
        .plan();
    let mut pyramid_options = RasterPyramidOptions::new(RasterPlacement::Explicit(pixel_to_source));
    pyramid_options.tile_format = options.tile_format;
    // A 1-bit gray source with no alpha is white or black and nothing between, so it keeps one
    // bit per pixel (PNG only: a JPEG tile is eight-bit).
    if decoded.bilevel
        && decoded.channels == Channels::Gray
        && options.tile_format == TileFormat::Png
    {
        pyramid_options.bilevel_threshold = Some(128);
        notes.push(PdfImageNote::Bilevel);
    }
    let mut pyramid =
        produce_raster_pyramid(&raster, &plan, &pyramid_options).map_err(PdfImageError::Pyramid)?;
    pyramid.visible_box = clips.visible_box;

    // Band: before every painted path is the underlay, after every one the overlay.
    let seq = image.provenance.sequence;
    let before = painted.iter().filter(|s| **s < seq).count();
    if before == 0 {
        pyramid.band = RasterBand::Underlay;
    } else if before == painted.len() {
        pyramid.band = RasterBand::Overlay;
    } else {
        pyramid.band = RasterBand::Underlay;
        notes.push(PdfImageNote::Interleaved);
    }
    Ok((pyramid, notes, decoded.filter))
}

/// `pixel -> page` composed with `page -> source`, `-0.0` normalised, and checked against NP-19:
/// each coefficient finite, the four linear entries at most 2^20, and the map invertible.
fn placement(to_page: [f64; 6], page_to_source: [f64; 6]) -> Result<[f64; 6], PdfImageError> {
    let [a1, b1, c1, d1, x1, y1] = to_page;
    let [a2, b2, c2, d2, x2, y2] = page_to_source;
    let m = [
        a2 * a1 + b2 * c1,
        a2 * b1 + b2 * d1,
        c2 * a1 + d2 * c1,
        c2 * b1 + d2 * d1,
        a2 * x1 + b2 * y1 + x2,
        c2 * x1 + d2 * y1 + y2,
    ];
    if !m.iter().all(|v| v.is_finite()) {
        return Err(PdfImageError::InvalidPlacement(
            "a coefficient is not finite",
        ));
    }
    if m[..4].iter().any(|v| v.abs() > f64::from(1u32 << 20)) {
        return Err(PdfImageError::InvalidPlacement(
            "a linear entry is larger than 2^20",
        ));
    }
    let det = m[0] * m[3] - m[1] * m[2];
    if det == 0.0 || !det.is_finite() {
        return Err(PdfImageError::InvalidPlacement("the map is singular"));
    }
    Ok(m.map(|v| if v == 0.0 { 0.0 } else { v }))
}

// ---- clips ----

/// What the clips in force leave of an image, in pixel space.
struct ClipPlan {
    /// `[x0 y0 x1 y1]` inside the image.
    visible_box: [u32; 4],
    /// Clips that are not a pixel-aligned rectangle: their edges and winding rule.
    polygons: Vec<(Vec<[f64; 4]>, Rule)>,
    width: u32,
    height: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Rule {
    NonZero,
    EvenOdd,
}

fn clip_plan(
    page: &PdfVectorPage,
    image: &PdfImagePlacement,
    to_page: [f64; 6],
    w: u32,
    h: u32,
) -> Result<ClipPlan, PdfImageError> {
    let [a, b, c, d, tx, ty] = to_page;
    let det = a * d - b * c;
    let to_pixel = |p: [f64; 2]| -> [f64; 2] {
        let (x, y) = (p[0] - tx, p[1] - ty);
        [(d * x - b * y) / det, (-c * x + a * y) / det]
    };
    let mut bx = [0.0, 0.0, f64::from(w), f64::from(h)];
    let mut polygons = Vec::new();
    for seq in &image.active_clips {
        // A sequence is the entity's index in the page, so this is a lookup and not a search.
        let Some(entity) = usize::try_from(*seq)
            .ok()
            .and_then(|i| page.entities.get(i))
            .filter(|e| e.provenance.sequence == *seq)
        else {
            continue;
        };
        let rule = match entity.clip {
            Some(super::pdf_vector::PdfFillRule::EvenOdd) => Rule::EvenOdd,
            _ => Rule::NonZero,
        };
        if let Some(r) = exact_rect(entity) {
            let p = [
                to_pixel(r.min),
                to_pixel([r.max[0], r.min[1]]),
                to_pixel(r.max),
                to_pixel([r.min[0], r.max[1]]),
            ];
            let (x0, x1) = min_max(p.iter().map(|q| q[0]));
            let (y0, y1) = min_max(p.iter().map(|q| q[1]));
            let tol = 1e-7;
            let aligned = p.iter().all(|q| {
                ((q[0] - x0).abs() < tol || (q[0] - x1).abs() < tol)
                    && ((q[1] - y0).abs() < tol || (q[1] - y1).abs() < tol)
            });
            let integral = [x0, y0, x1, y1].iter().all(|v| (v - v.round()).abs() < tol);
            if aligned && integral {
                intersect(&mut bx, [x0.round(), y0.round(), x1.round(), y1.round()]);
                continue;
            }
        }
        let edges = edges_in_pixels(entity, &to_pixel);
        if edges.is_empty() {
            // A clip with no area leaves nothing.
            return Err(PdfImageError::FullyClipped);
        }
        let (x0, x1) = min_max(edges.iter().flat_map(|e| [e[0], e[2]]));
        let (y0, y1) = min_max(edges.iter().flat_map(|e| [e[1], e[3]]));
        intersect(&mut bx, [x0.floor(), y0.floor(), x1.ceil(), y1.ceil()]);
        polygons.push((edges, rule));
    }
    if bx[0] >= bx[2] || bx[1] >= bx[3] {
        return Err(PdfImageError::FullyClipped);
    }
    Ok(ClipPlan {
        visible_box: bx.map(|v| v as u32),
        polygons,
        width: w,
        height: h,
    })
}

fn min_max(values: impl Iterator<Item = f64>) -> (f64, f64) {
    values.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
        (lo.min(v), hi.max(v))
    })
}

fn intersect(bx: &mut [f64; 4], other: [f64; 4]) {
    bx[0] = bx[0].max(other[0]);
    bx[1] = bx[1].max(other[1]);
    bx[2] = bx[2].min(other[2]);
    bx[3] = bx[3].min(other[3]);
}

/// The closed outline of a clip entity as straight edges in pixel space. Curves are cut into 16
/// chords, which is far below a pixel for any clip that fits an image.
fn edges_in_pixels(
    entity: &PdfPathEntity,
    to_pixel: &dyn Fn([f64; 2]) -> [f64; 2],
) -> Vec<[f64; 4]> {
    let mut edges = Vec::new();
    let (mut start, mut cur): (Option<[f64; 2]>, Option<[f64; 2]>) = (None, None);
    let close = |edges: &mut Vec<[f64; 4]>, start: Option<[f64; 2]>, cur: Option<[f64; 2]>| {
        if let (Some(s), Some(c)) = (start, cur)
            && s != c
        {
            edges.push([c[0], c[1], s[0], s[1]]);
        }
    };
    for seg in &entity.segments {
        match seg {
            PdfPathSegment::MoveTo(p) => {
                close(&mut edges, start, cur);
                let q = to_pixel(*p);
                (start, cur) = (Some(q), Some(q));
            }
            PdfPathSegment::LineTo(p) => {
                let q = to_pixel(*p);
                if let Some(c) = cur {
                    edges.push([c[0], c[1], q[0], q[1]]);
                }
                cur = Some(q);
            }
            PdfPathSegment::CurveTo { c1, c2, end } => {
                let (p1, p2, p3) = (to_pixel(*c1), to_pixel(*c2), to_pixel(*end));
                let Some(p0) = cur else { continue };
                let mut last = p0;
                for i in 1..=16 {
                    let t = f64::from(i) / 16.0;
                    let u = 1.0 - t;
                    let at = |k: usize| {
                        u * u * u * p0[k]
                            + 3.0 * u * u * t * p1[k]
                            + 3.0 * u * t * t * p2[k]
                            + t * t * t * p3[k]
                    };
                    let q = [at(0), at(1)];
                    edges.push([last[0], last[1], q[0], q[1]]);
                    last = q;
                }
                cur = Some(p3);
            }
            PdfPathSegment::Close => {
                close(&mut edges, start, cur);
                cur = start;
            }
        }
    }
    close(&mut edges, start, cur);
    edges
}

/// Write the clips into the alpha channel: a pixel is kept when its centre is inside every
/// polygon clip (by the clip's own winding rule) and inside the visible box.
fn apply_alpha(d: Decoded, clips: &ClipPlan) -> Decoded {
    let (w, h) = (clips.width as usize, clips.height as usize);
    let n = d.channels.count();
    let mut keep = vec![0u8; w * h];
    let [x0, y0, x1, y1] = clips.visible_box.map(|v| v as usize);
    for row in keep.chunks_mut(w).take(y1).skip(y0) {
        row[x0..x1].fill(1);
    }
    let mut cover = vec![0u8; w];
    for (edges, rule) in &clips.polygons {
        for (y, row) in keep.chunks_mut(w).enumerate().take(y1).skip(y0) {
            cover.fill(0);
            let yc = y as f64 + 0.5;
            let mut cross: Vec<(f64, i32)> = edges
                .iter()
                .filter(|e| (e[1] <= yc && yc < e[3]) || (e[3] <= yc && yc < e[1]))
                .map(|e| {
                    let x = e[0] + (yc - e[1]) * (e[2] - e[0]) / (e[3] - e[1]);
                    (x, if e[3] > e[1] { 1 } else { -1 })
                })
                .collect();
            cross.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut winding = 0;
            for (i, pair) in cross.windows(2).enumerate() {
                winding += pair[0].1;
                let inside = match rule {
                    Rule::NonZero => winding != 0,
                    Rule::EvenOdd => i % 2 == 0,
                };
                if inside {
                    // Pixels whose centre lies in [xa, xb).
                    let from = (pair[0].0 - 0.5).ceil().max(0.0) as usize;
                    let to = ((pair[1].0 - 0.5).ceil().max(0.0) as usize).min(w);
                    if from < to {
                        cover[from..to].fill(1);
                    }
                }
            }
            for (k, c) in row.iter_mut().zip(&cover) {
                *k &= *c;
            }
        }
        // Rows outside the box were never computed and stay 0.
    }
    let mut data = Vec::with_capacity(w * h * 4);
    for (i, px) in d.data.chunks(n).enumerate() {
        let a = if keep[i] == 1 { 255 } else { 0 };
        match d.channels {
            Channels::Gray => data.extend_from_slice(&[px[0], px[0], px[0], a]),
            Channels::Rgb => data.extend_from_slice(&[px[0], px[1], px[2], a]),
            Channels::Rgba => data.extend_from_slice(&[px[0], px[1], px[2], px[3].min(a)]),
        }
    }
    Decoded {
        channels: Channels::Rgba,
        data,
        ..d
    }
}
