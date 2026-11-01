//! `JBIG2Decode`: embedded organisation (T.88 Annex D.3) through `hayro-jbig2`, behind the
//! non-default `jbig2` feature (#1365).
//!
//! The output is 1 bit per pixel, rows padded to bytes, with **0 as black**: JBIG2 calls black 1
//! and PDF's `DeviceGray` calls it 0, so every bit is inverted on the way out, and a
//! `/Decode [1 0]` on the image flips it back in the colour pipeline as for any other 1-bit image.
//!
//! `hayro-jbig2` 0.3.1 has no lenient mode to switch off: a stream it cannot decode in full is an
//! error, so a picture is never returned part way. Newer, unreleased code adds
//! `DecodeSettings { strict }`, and when that ships it is passed `true` here.
//!
//! Limits: the page size is read from the stream and compared with `/Width` and `/Height`
//! before anything is decoded, and the caller has already held those to `max_pixels`, so the
//! page bitmap the decoder allocates is bounded by the same budget as every other codec's.
//! The globals stream is counted against the per-stream cap by the extraction that kept it.

#[cfg(feature = "jbig2")]
use crate::pvec::pdf_image::FilterError;
use crate::pvec::pdf_image::PdfImageError;
use crate::pvec::pdf_image_decode::{FilterInput, FilterOutput};

#[cfg(not(feature = "jbig2"))]
pub(in crate::pvec) fn decode(input: FilterInput<'_>) -> Result<FilterOutput, PdfImageError> {
    Err(PdfImageError::FilterNotEnabled {
        filter: input.filter.to_string(),
        feature: "jbig2",
    })
}

/// Collects the decoder's pixels as PDF 1-bit rows, 0 black.
#[cfg(feature = "jbig2")]
struct Rows {
    row_bytes: usize,
    height: usize,
    data: Vec<u8>,
    row: Vec<bool>,
}

#[cfg(feature = "jbig2")]
impl hayro_jbig2::Decoder for Rows {
    fn push_pixel(&mut self, black: bool) {
        self.row.push(black);
    }

    fn push_pixel_chunk(&mut self, black: bool, chunk_count: u32) {
        // A hostile stream cannot make a row longer than the page the caller already budgeted,
        // but stop growing past one row plus a chunk all the same.
        let room = (self.row_bytes * 8 + 8).saturating_sub(self.row.len());
        let n = (chunk_count as usize * 8).min(room);
        self.row.extend(std::iter::repeat_n(black, n));
    }

    fn next_line(&mut self) {
        if self.data.len() / self.row_bytes.max(1) < self.height {
            let mut bytes = vec![0u8; self.row_bytes];
            for (x, black) in self.row.iter().take(self.row_bytes * 8).enumerate() {
                // 0 is black, so a white pixel sets the bit.
                if !black {
                    bytes[x / 8] |= 0x80 >> (x % 8);
                }
            }
            // The padding bits past the last pixel read as white.
            self.data.extend_from_slice(&bytes);
        }
        self.row.clear();
    }
}

#[cfg(feature = "jbig2")]
pub(in crate::pvec) fn decode(input: FilterInput<'_>) -> Result<FilterOutput, PdfImageError> {
    let fail = |source: FilterError| PdfImageError::Filter {
        filter: input.filter.to_string(),
        source,
    };
    let image = hayro_jbig2::Image::new_embedded(input.data, input.globals)
        .map_err(|e| fail(FilterError::Jbig2(e.to_string())))?;
    if (image.width(), image.height()) != (input.width, input.height) {
        return Err(fail(FilterError::Parms(format!(
            "the JBIG2 page is {}x{} but the image says {}x{}",
            image.width(),
            image.height(),
            input.width,
            input.height
        ))));
    }
    let row_bytes = (input.width as usize).div_ceil(8);
    let height = input.height as usize;
    let mut rows = Rows {
        row_bytes,
        height,
        data: Vec::with_capacity(row_bytes.saturating_mul(height)),
        row: Vec::new(),
    };
    image
        .decode(&mut rows)
        .map_err(|e| fail(FilterError::Jbig2(e.to_string())))?;
    let need = row_bytes * height;
    if rows.data.len() < need {
        return Err(fail(FilterError::ShortOutput {
            got: rows.data.len(),
            need,
        }));
    }
    Ok(FilterOutput::Packed {
        data: rows.data,
        bpc: 1,
    })
}
