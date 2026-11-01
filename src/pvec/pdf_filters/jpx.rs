//! `JPXDecode` through [`crate::jp2k`] (issue #1364): the `jp2k` feature's `hayro-jpeg2000`, so
//! nothing new enters the dependency graph. Without the feature the body is the typed
//! `FilterNotEnabled` naming it, the way `jp2k.rs` itself carries a `FeatureNotEnabled` body.
//!
//! The codestream says how many components it has and what they mean, so a dictionary
//! `/ColorSpace` that disagrees in component count loses to it, as for `DCTDecode`, and with no
//! `/ColorSpace` at all the codestream's decides (`space_from_codestream`). `/Decode` is ignored
//! for a non-Indexed space and `/SMaskInData` other than 0 is refused by name; both are the
//! caller's, in `pdf_image_decode.rs`, because they read the image dictionary. A JPX image is
//! reported as a re-encode of a lossy source like a JPEG is (a reversible codestream is still
//! re-tiled, and the decoder does not say which it was).

use crate::pvec::pdf_image::PdfImageError;
use crate::pvec::pdf_image_decode::{FilterInput, FilterOutput};

#[cfg(not(feature = "jp2k"))]
pub(in crate::pvec) fn decode(input: FilterInput<'_>) -> Result<FilterOutput, PdfImageError> {
    Err(PdfImageError::FilterNotEnabled {
        filter: input.filter.to_string(),
        feature: "jp2k",
    })
}

#[cfg(feature = "jp2k")]
pub(in crate::pvec) fn decode(input: FilterInput<'_>) -> Result<FilterOutput, PdfImageError> {
    use crate::conversion::Interpretation;
    use crate::pixel::PixelFormat;
    use crate::pvec::pdf_image::{FilterError, PdfImageNote};
    use crate::source::SourceError;

    let refuse = |source: FilterError| PdfImageError::Filter {
        filter: input.filter.to_string(),
        source,
    };
    let raster = crate::jp2k::decode_jp2k(input.data, input.budget).map_err(|e| match e {
        SourceError::Jp2k(j) => refuse(FilterError::Jp2k(j)),
        other => refuse(FilterError::Source(other)),
    })?;
    if (raster.width(), raster.height()) != (input.width, input.height) {
        return Err(PdfImageError::Decode(format!(
            "the JPEG 2000 codestream is {}x{} but the image says {}x{}",
            raster.width(),
            raster.height(),
            input.width,
            input.height
        )));
    }
    // Colour channels kept and the carrier's samples per pixel. Alpha is dropped here, because
    // `/SMaskInData` 0 (the only value that gets this far) says it is not applied. Four
    // components are CMYK when the codestream's `colr` says so or the dictionary names a
    // four-component space, and RGB plus alpha otherwise.
    let cmyk = raster.interpretation() == Interpretation::Cmyk || input.components == Some(4);
    let (kept, per_pixel, sixteen): (u8, usize, bool) = match raster.format() {
        PixelFormat::Gray8 => (1, 1, false),
        PixelFormat::Rgb8 => (3, 3, false),
        PixelFormat::Rgba8 if cmyk => (4, 4, false),
        PixelFormat::Rgba8 => (3, 4, false),
        PixelFormat::Gray16 => (1, 1, true),
        PixelFormat::Rgb16 => (3, 3, true),
        PixelFormat::Rgba16 if cmyk => (4, 4, true),
        PixelFormat::Rgba16 => (3, 4, true),
        PixelFormat::Multi8(n) if n.get() == 2 => (1, 2, false),
        PixelFormat::Multi8(n) if n.get() == 4 => (4, 4, false),
        PixelFormat::Multi16(n) if n.get() == 2 => (1, 2, true),
        PixelFormat::Multi16(n) if n.get() == 4 => (4, 4, true),
        other => {
            return Err(PdfImageError::Decode(format!(
                "a JPEG 2000 codestream decoded to {other:?}, not gray, RGB or CMYK"
            )));
        }
    };
    let mut notes = vec![PdfImageNote::ReEncodedLossy];
    let samples: Vec<u8> = if sixteen {
        notes.push(PdfImageNote::BitDepthReduced);
        // Native-endian u16 samples; the high byte is the 8-bit value.
        raster
            .data()
            .chunks_exact(2 * per_pixel)
            .flat_map(|px| {
                (0..usize::from(kept))
                    .map(move |c| u16::from_ne_bytes([px[2 * c], px[2 * c + 1]]).to_be_bytes()[0])
            })
            .collect()
    } else {
        raster
            .data()
            .chunks_exact(per_pixel)
            .flat_map(|px| px[..usize::from(kept)].iter().copied())
            .collect()
    };
    // A profile the bake cannot apply: the dictionary names no space, so the codestream's ICC
    // one would have decided, and the samples are read as the device space of that many
    // components. CMYK is already reported by the space itself.
    if input.components.is_none() && kept != 4 && raster.icc_profile().is_some() {
        notes.push(PdfImageNote::ColourSpaceApproximated);
    }
    Ok(FilterOutput::Samples {
        data: samples,
        channels: kept,
        space_from_codestream: input.components.is_none(),
        notes,
    })
}
