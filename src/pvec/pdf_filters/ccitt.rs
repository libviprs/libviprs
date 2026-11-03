//! `CCITTFaxDecode` through `hayro-ccitt` (issue #1363), unconditional: no feature.
//!
//! The codec hands back the colour changes of each row; [`Rows`] packs them into one bit per
//! sample, rows padded to bytes, which is what the colour pipeline reads for any 1-bit gray image.
//! `/BlackIs1 false` (the default) makes a black pixel a 0 bit, the DeviceGray convention, so a
//! plain scan reads black on white with no `/Decode`. The output is allocated once at `ceil(width /
//! 8) * height`, and the pixel budget has already bounded `width * height`, so a hostile stream
//! cannot grow it.
//!
//! `/Columns` (default 1728) and `/Rows` have to agree with `/Width` and `/Height`: a mismatch
//! is [`FilterError::Parms`], never a silent resize.

use hayro_ccitt::{DecodeError, DecodeSettings, Decoder, DecoderContext, EncodingMode};
use lopdf::Dictionary;

use crate::pvec::pdf_image::{FilterError, PdfImageError};
use crate::pvec::pdf_image_decode::{FilterInput, FilterOutput};

const FILTER: &str = "CCITTFaxDecode";

fn refuse(source: FilterError) -> PdfImageError {
    PdfImageError::Filter {
        filter: FILTER.to_string(),
        source,
    }
}

fn parms_err(detail: &str) -> PdfImageError {
    refuse(FilterError::Parms(detail.to_string()))
}

/// The rows of the codec, packed. The codec's [`Decoder`] trait needs a named type.
struct Rows {
    data: Vec<u8>,
    row_bytes: usize,
    width: usize,
    height: usize,
    /// Rows finished.
    y: usize,
    /// `/BlackIs1`: a black pixel is a 1 bit.
    black_is_1: bool,
}

impl Decoder for Rows {
    fn push_row(&mut self, changes: &[u32]) {
        if self.y < self.height {
            let width = self.width;
            let row = &mut self.data[self.y * self.row_bytes..(self.y + 1) * self.row_bytes];
            // The row starts white and `changes` flips it. A white pixel is a 1 bit unless
            // `/BlackIs1` turns the convention round, and the padding past the width stays 0.
            let mut set = |from: usize, to: usize| {
                for x in from..to.min(width) {
                    row[x / 8] |= 0x80 >> (x % 8);
                }
            };
            let mut at = 0;
            for run in changes.chunks(2) {
                let start = run[0] as usize;
                let end = run.get(1).map_or(width, |&e| e as usize);
                if self.black_is_1 {
                    set(start, end);
                } else {
                    set(at, start);
                }
                at = end;
            }
            if !self.black_is_1 {
                set(at, width);
            }
        }
        self.y = (self.y + 1).min(self.height);
    }
}

fn int(parms: Option<&Dictionary>, key: &[u8], default: i64) -> Result<i64, PdfImageError> {
    match parms.and_then(|d| d.get(key).ok()) {
        None => Ok(default),
        Some(o) => o.as_i64().map_err(|_| {
            parms_err(&format!(
                "/{} is not an integer",
                String::from_utf8_lossy(key)
            ))
        }),
    }
}

fn flag(parms: Option<&Dictionary>, key: &[u8], default: bool) -> Result<bool, PdfImageError> {
    match parms.and_then(|d| d.get(key).ok()) {
        None => Ok(default),
        Some(o) => o.as_bool().map_err(|_| {
            parms_err(&format!(
                "/{} is not a boolean",
                String::from_utf8_lossy(key)
            ))
        }),
    }
}

pub(in crate::pvec) fn decode(input: FilterInput<'_>) -> Result<FilterOutput, PdfImageError> {
    let parms = input.parms;
    if input.bpc.is_some_and(|b| b != 1) || input.components.is_some_and(|n| n != 1) {
        return Err(parms_err("a fax image is one bit and one component"));
    }
    let k = int(parms, b"K", 0)?;
    let columns = int(parms, b"Columns", 1728)?;
    let rows = int(parms, b"Rows", 0)?;
    if columns != i64::from(input.width) {
        return Err(parms_err(&format!(
            "/Columns {columns} is not /Width {}",
            input.width
        )));
    }
    if rows != 0 && rows != i64::from(input.height) {
        return Err(parms_err(&format!(
            "/Rows {rows} is not /Height {}",
            input.height
        )));
    }
    let encoding = match k {
        ..=-1 => EncodingMode::Group4,
        0 => EncodingMode::Group3_1D,
        positive => {
            u32::try_from(positive).map_err(|_| parms_err("/K is out of range"))?;
            EncodingMode::Group3_2D
        }
    };
    let settings = DecodeSettings {
        columns: input.width,
        rows: input.height,
        end_of_block: flag(parms, b"EndOfBlock", true)?,
        end_of_line: flag(parms, b"EndOfLine", false)?,
        rows_are_byte_aligned: flag(parms, b"EncodedByteAlign", false)?,
        encoding,
        // Lenient, as before: a scan with a damaged tail is a short scan, not a refusal.
        strict: false,
    };
    let width = input.width as usize;
    let height = input.height as usize;
    let row_bytes = width.div_ceil(8);
    let need = row_bytes.saturating_mul(height);
    let mut sink = Rows {
        data: vec![0; need],
        row_bytes,
        width,
        height,
        y: 0,
        black_is_1: flag(parms, b"BlackIs1", false)?,
    };
    let result = hayro_ccitt::decode(input.data, &mut sink, &mut DecoderContext::new(settings));
    // Rows that arrived before a late error still count: a scan with a bad tail is a short scan.
    if sink.y < height {
        return Err(refuse(match result {
            Err(e) if e != DecodeError::UnexpectedEof => FilterError::Ccitt(e.to_string()),
            _ => FilterError::ShortOutput {
                got: sink.y * row_bytes,
                need,
            },
        }));
    }
    Ok(FilterOutput::Packed {
        data: sink.data,
        bpc: 1,
    })
}
