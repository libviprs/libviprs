//! `JPXDecode`: a stub until the codec lands. JPEG 2000 (#1364 decodes it through crate::jp2k).

use crate::pvec::pdf_image::PdfImageError;
use crate::pvec::pdf_image_decode::{FilterInput, FilterOutput};

pub(in crate::pvec) fn decode(input: FilterInput<'_>) -> Result<FilterOutput, PdfImageError> {
    Err(PdfImageError::FilterNotEnabled {
        filter: input.filter.to_string(),
        feature: "jp2k",
    })
}
