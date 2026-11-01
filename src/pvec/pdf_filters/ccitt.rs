//! `CCITTFaxDecode`: a stub until the codec lands. CCITT (#1363 decodes it through hayro-ccitt, with no feature).

use crate::pvec::pdf_image::PdfImageError;
use crate::pvec::pdf_image_decode::{FilterInput, FilterOutput};

pub(in crate::pvec) fn decode(input: FilterInput<'_>) -> Result<FilterOutput, PdfImageError> {
    Err(PdfImageError::FilterNotEnabled {
        filter: input.filter.to_string(),
        feature: "ccitt",
    })
}
