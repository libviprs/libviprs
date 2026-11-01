//! `DCTDecode`: a JPEG, decoded to eight-bit gray or RGB samples.

use crate::pixel::PixelFormat;
use crate::pvec::pdf_image::{FilterError, PdfImageError, PdfImageNote};
use crate::pvec::pdf_image_decode::{FilterInput, FilterOutput};

pub(in crate::pvec) fn decode(input: FilterInput<'_>) -> Result<FilterOutput, PdfImageError> {
    let (width, height) = (input.width, input.height);
    let jpeg = crate::source::decode_bytes_with_limits(input.data, input.budget).map_err(|e| {
        PdfImageError::Filter {
            filter: input.filter.to_string(),
            source: FilterError::Dct(e),
        }
    })?;
    if (jpeg.width(), jpeg.height()) != (width, height) {
        return Err(PdfImageError::Decode(format!(
            "the JPEG is {}x{} but the image says {width}x{height}",
            jpeg.width(),
            jpeg.height()
        )));
    }
    let channels = match jpeg.format() {
        PixelFormat::Gray8 => 1,
        PixelFormat::Rgb8 => 3,
        other => {
            return Err(PdfImageError::Decode(format!(
                "a JPEG decoded to {other:?}, not gray or RGB"
            )));
        }
    };
    Ok(FilterOutput::Samples {
        data: jpeg.data().to_vec(),
        channels,
        space_from_codestream: false,
        notes: vec![PdfImageNote::ReEncodedLossy],
    })
}
