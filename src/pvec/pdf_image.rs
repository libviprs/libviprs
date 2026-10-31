//! PDF image placements: decode, bake and cut the images of a page into PVec rasters
//! (issue #1332, RL2.1 of EPIC R-L, ADR 0018 in libviprs/pvec).
//!
//! Skeleton: the types are final, the bake is the next commit.

use std::collections::BTreeMap;

use thiserror::Error;

use super::pdf_vector::PdfVectorPage;
use crate::raster_pvec::pyramid::{RasterPyramid, RasterPyramidError};
use crate::sink::TileFormat;

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
        }
    }
}

/// Why an image was not written.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PdfImageError {
    /// The decode did not happen yet.
    #[error("image decode failed: {0}")]
    Decode(String),
    /// A filter the crate does not decode (`JPXDecode`, `JBIG2Decode`, `CCITTFaxDecode`...).
    #[error("unsupported filter {0}")]
    UnsupportedFilter(String),
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

impl PdfImageError {
    /// The stable reason the report counts it under.
    #[must_use]
    pub fn reason(&self) -> String {
        match self {
            Self::Decode(_) => "decode-failed".to_string(),
            Self::UnsupportedFilter(n) => format!("unsupported-filter:{n}"),
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

/// Bake every image of `page`.
#[must_use]
pub fn bake_page_images(page: &PdfVectorPage, _options: &PdfImageOptions) -> PdfImageBake {
    let mut report = PdfImageReport {
        images: page.images.len() as u64,
        ..PdfImageReport::default()
    };
    let refused = (0..page.images.len())
        .map(|index| PdfRefusedImage {
            index,
            error: PdfImageError::Decode("the bake is not written yet".to_string()),
        })
        .collect::<Vec<_>>();
    report
        .refused
        .insert("decode-failed".to_string(), refused.len() as u64);
    PdfImageBake {
        written: Vec::new(),
        refused,
        report,
    }
}
