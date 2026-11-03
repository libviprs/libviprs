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
//!
//! `hayro-jbig2` 0.3.1 bounds each bitmap's sides (65535) and the symbol count (65535) but not
//! what they multiply to, and its arithmetic decoder keeps producing symbols after the data has
//! run out, so a 134 byte stream found by `fuzz_pvec_pdf_image` made it allocate over 2 GB in
//! symbol bitmaps. [`prescan`] closes the part that can be closed from outside: a region larger
//! than the page is refused, and a symbol dictionary may not declare more new symbols than its
//! own data has bits. The size of each symbol is still the decoder's to bound, and it does not:
//! the same fuzz run found a stream that spends about 400 seconds in one symbol dictionary (a
//! symbol up to 65535 by 65535 pixels, decoded pixel by pixel). That is the open finding behind
//! `jbig2` not being a default feature, and it is not something this module can bound from outside.
//! The other fuzz findings (an `export_symbols` loop that never ends, a halftone grid of billions
//! of cells, empty height classes) are in arithmetic coded data too. All are reported upstream
//! (`LaurenzV/hayro` #1700, #1701, #1550, #1598) and kept as ignored tests.

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

/// Refuse what a segment header says before the decoder allocates for it (T.88 7.2 and 7.4).
///
/// Walks the sequential segment headers of `stream`. A region segment (text 4 and 6 to 7, halftone
/// 20 and 22 to 23, generic 36 and 38 to 39, refinement 40 and 42 to 43, so the intermediate forms too) that is wider or taller than the page, or a
/// symbol dictionary (type 0) whose `SDNUMNEWSYMS` is over eight times its data length, is an
/// error. A segment of unknown length ends the walk: only an immediate generic region may have
/// one, and its header is read first.
#[cfg(feature = "jbig2")]
fn prescan(stream: &[u8], width: u32, height: u32) -> Result<(), String> {
    let be32 = |at: usize| -> Option<u32> {
        Some(u32::from_be_bytes(stream.get(at..at + 4)?.try_into().ok()?))
    };
    let mut at = 0;
    while at < stream.len() {
        let truncated = || "a segment header runs past the data".to_string();
        let number = be32(at).ok_or_else(truncated)?;
        let flags = *stream.get(at + 4).ok_or_else(truncated)?;
        at += 5;
        let first = *stream.get(at).ok_or_else(truncated)?;
        let referred = if first >> 5 == 7 {
            let n = (be32(at).ok_or_else(truncated)? & 0x1FFF_FFFF) as usize;
            at += 4 + (n + 1).div_ceil(8);
            n
        } else {
            at += 1;
            usize::from(first >> 5)
        };
        let width_of_number = if number <= 256 {
            1
        } else if number <= 65536 {
            2
        } else {
            4
        };
        at = at.saturating_add(referred.saturating_mul(width_of_number));
        at += if flags & 0x40 != 0 { 4 } else { 1 };
        let length = be32(at).ok_or_else(truncated)?;
        at += 4;
        let unknown = length == 0xFFFF_FFFF;
        let end = if unknown {
            stream.len()
        } else {
            at.saturating_add(length as usize)
        };
        let data = stream
            .get(at..end.min(stream.len()))
            .ok_or_else(truncated)?;
        match flags & 0x3F {
            4 | 6 | 7 | 20 | 22 | 23 | 36 | 38 | 39 | 40 | 42 | 43 => {
                let w = be32(at).ok_or_else(truncated)?;
                let h = be32(at + 4).ok_or_else(truncated)?;
                if w > width || h > height {
                    return Err(format!("a {w}x{h} region on a {width}x{height} page"));
                }
            }
            0 => {
                let sd = u16::from_be_bytes([
                    *data.first().ok_or_else(truncated)?,
                    *data.get(1).ok_or_else(truncated)?,
                ]);
                let huffman = sd & 1 != 0;
                let refagg = sd & 2 != 0;
                let template = (sd >> 10) & 3;
                let rtemplate = (sd >> 12) & 1;
                let mut off = 2;
                if !huffman {
                    off += if template == 0 { 8 } else { 2 };
                }
                if refagg && rtemplate == 0 {
                    off += 4;
                }
                let symbols = u32::from_be_bytes(
                    data.get(off + 4..off + 8)
                        .ok_or_else(truncated)?
                        .try_into()
                        .map_err(|_| truncated())?,
                );
                if u64::from(symbols) > data.len() as u64 * 8 {
                    return Err(format!(
                        "a symbol dictionary of {symbols} symbols in {} bytes",
                        data.len()
                    ));
                }
            }
            _ => {}
        }
        if unknown {
            break;
        }
        at = end;
    }
    Ok(())
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
    for stream in [input.globals, Some(input.data)].into_iter().flatten() {
        prescan(stream, input.width, input.height).map_err(|m| fail(FilterError::Jbig2(m)))?;
    }
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
