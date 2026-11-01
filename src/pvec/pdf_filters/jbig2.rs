//! `JBIG2Decode`: a stub until the codec lands. JBIG2 (#1365 decodes it through hayro-jbig2 behind the jbig2 feature).

use crate::pvec::pdf_image::PdfImageError;
use crate::pvec::pdf_image_decode::{FilterInput, FilterOutput};

pub(in crate::pvec) fn decode(input: FilterInput<'_>) -> Result<FilterOutput, PdfImageError> {
    Err(PdfImageError::FilterNotEnabled {
        filter: input.filter.to_string(),
        feature: "jbig2",
    })
}
