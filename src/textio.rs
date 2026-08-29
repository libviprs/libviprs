//! Pure-Rust text and tabular raster codecs: the libvips `matrix` and `csv`
//! text formats and the Netpbm `ppm`/`pgm`/`pbm` family.
//!
//! These formats carry pixel values as plain text or as a tiny binary
//! Netpbm blob, so they need no external codec crate. Every encoder here has
//! a decoder that round-trips it losslessly; the Netpbm decoder reads more
//! than the Netpbm encoder writes, which is deliberate and is set out below:
//!
//! | format | encode | decode | reload as |
//! |---|---|---|---|
//! | libvips `matrix` | [`Raster::matrix_save`] | [`Raster::matrix_load`] | single-band float |
//! | CSV | [`Raster::csv_save`] | [`Raster::csv_load`] | single-band float |
//! | Netpbm PPM/PGM | [`Raster::ppm_save`] / [`Raster::encode_ppm`] | [`Raster::ppm_load`] | 1- or 3-band `uchar`/`ushort` |
//! | Netpbm PBM (`P1`, `P4`) | nothing writes these | [`Raster::ppm_load`] | 1-band `uchar`, `1` black and `0` white |
//!
//! The decoders are associated functions on [`Raster`] (`Raster::matrix_load`,
//! `Raster::csv_load`, `Raster::ppm_load`), so a caller reaches them through the
//! type that carries the encoders, with no free-function import.
//!
//! ## Matrix text format
//!
//! The libvips `matrixsave`/`matrixload` format: a header line holding the
//! `width` and `height` (libvips also admits an optional `scale offset` pair,
//! which this reader tolerates and ignores), followed by `height` rows of
//! `width` whitespace-separated numbers. libvips reports it as a `double`,
//! one-band image; the pipeline's float carrier is `f32`, matching
//! [`Raster::from_matrix`], so the values are stored as `FloatF32(1)`.
//!
//! ## CSV
//!
//! A comma-separated grid with no header: the column count of the first row
//! fixes the width and the row count fixes the height. Like `matrix`, CSV is
//! a one-band numeric format and reloads as `FloatF32(1)`.
//!
//! ## Netpbm PPM/PGM/PBM
//!
//! [`Raster::ppm_load`] reads six ASCII/binary variants: `P1` (ASCII bitmap),
//! `P2` (ASCII grey), `P3` (ASCII RGB), `P4` (binary bitmap), `P5` (binary
//! grey) and `P6` (binary RGB), including `#` header comments. A `maxval` of
//! `255` or less decodes to an 8-bit raster; a larger `maxval` (up to `65535`)
//! decodes to 16-bit, with binary samples read most-significant-byte-first as
//! the Netpbm specification requires. [`Raster::ppm_save`] and
//! [`Raster::encode_ppm`] emit the binary form: `P5` for a one-band raster and
//! `P6` for a three-band one.
//!
//! ## The bitmap forms are read and not written, and that is the decision
//!
//! `P1` and `P4` carry one bit per pixel and no `maxval` field at all. They
//! decode to `Gray8` with **`1` black and `0` white**, inverted against every
//! other format in this crate, which is why the polarity lives in one place
//! (`bitmap_ink`) with the four measured pixels on it. `P4` packs eight
//! pixels to a byte most significant bit first and pads **each row** out to a
//! whole byte, so a two-pixel-wide bitmap is one byte a row and six bits of
//! every row are padding rather than pixels.
//!
//! Nothing in this crate writes them, and that is a decision rather than an
//! omission.
//!
//! The property #882 established, and #910 restored, is that everything this
//! crate **writes** it can **read**. It has never been the converse: `P2` and
//! `P3` have decoded since #77 and no route has ever written either. The
//! decode set has always been strictly wider than the encode set, so widening
//! it by two more leaves the invariant that matters untouched.
//!
//! Writing `P4` would need a thresholding policy, and there is no oracle to
//! copy. Measured on the pinned vips 8.18.6, `vips ppmsave` to `.pbm`
//! thresholds at 128 (the grey ramp `0 1 2 127 128 129 254 255` writes `f0`,
//! so a sample under 128 is ink) and then **right-aligns the last partial byte
//! of every row**, where its own reader takes those bits from the top of the
//! byte:
//!
//! ```text
//! width  vips wrote  what vips then reads back out of its own file
//! 4      0c          all four white, for an input of 0 1 128 255
//! 5      1a          255 255 255 0 0, for an input of 0 0 255 0 255
//! 9      80 01       the ninth pixel white, where the input was black
//! 12     b2 0c       the last four white, where the input ended 255 255
//! ```
//!
//! **The reader is the correct half of that, checked rather than assumed.**
//! The same 5-wide picture hand-packed the way the spec says, left-aligned as
//! `d0`, reads back through vips as `0 0 255 0 255`, which is the grey input
//! thresholded. So `ppmload` is right and it is `ppmsave` that disagrees with
//! it, and this decoder matches the reader. Without that control the table
//! above has two readings and only one of them is true.
//!
//! Only a width that is a multiple of eight survives that. So matching the
//! oracle's bytes means writing files the oracle itself misreads, and writing
//! correct bytes means diverging from it; neither is parity. That is the same
//! ground #882 stood on when it left `.pbm` and `.pfm` out of the save routes,
//! that no encoder stands behind them.
//! `the_bitmap_forms_decode_and_nothing_here_writes_them` pins the asymmetry,
//! so the day an encoder does arrive it goes red and says so.

use crate::codec::{DecodeError, EncodeError};
use crate::pixel::{PixelFormat, SampleKind, read_sample_f64};
use crate::raster::Raster;
use std::io::{Error as IoError, ErrorKind};

/// Build a typed [`DecodeError`] for a malformed text or Netpbm input.
///
/// The decode surface reports through [`crate::source::SourceError`], which
/// has no dedicated text-format variant; a malformed body is an
/// [`std::io::ErrorKind::InvalidData`] I/O error, which converts into the
/// shared error so the caller gets a typed `Err` rather than a panic.
fn malformed(msg: impl Into<String>) -> DecodeError {
    IoError::new(ErrorKind::InvalidData, msg.into()).into()
}

/// The single-band float format used by the text codecs (`FloatF32(1)`).
fn float1() -> Result<PixelFormat, DecodeError> {
    PixelFormat::with_kind(1, SampleKind::F32)
        .ok_or_else(|| malformed("cannot build a single-band float format"))
}

/// Format the channel-0 sample at byte offset `off` as its shortest decimal.
///
/// Unsigned samples print as integers; float samples print with the shortest
/// representation that parses back to the same `f32`, so the text round-trips
/// losslessly.
fn fmt_sample(data: &[u8], off: usize, fmt: PixelFormat) -> String {
    let kind = fmt.kind();
    let v = read_sample_f64(data, kind, off);
    if kind.is_float() {
        // Back through `f32` so the shortest representation that parses to
        // the same `f32` is what gets written; going via `f64` would print
        // the widened decimal instead.
        (v as f32).to_string()
    } else {
        // Every integer kind's value is exact in `f64`, including `I32`'s
        // and `U32`'s, so this is the integer and not a rounding of it.
        (v as i64).to_string()
    }
}

impl Raster {
    /// Serialise as the libvips `matrix` text format.
    ///
    /// Writes a `width height` header line, then one row per image row with the
    /// band-0 sample of each pixel formatted as a space-separated decimal.
    /// `matrix` is a one-band numeric format, so only the first band is
    /// written; the inverse is [`Raster::matrix_load`].
    pub fn matrix_save(&self) -> Vec<u8> {
        let w = self.width() as usize;
        let h = self.height() as usize;
        let fmt = self.format();
        let bpp = fmt.bytes_per_pixel();
        let stride = self.stride();
        let data = self.data();

        let mut out = String::new();
        out.push_str(&self.width().to_string());
        out.push(' ');
        out.push_str(&self.height().to_string());
        out.push('\n');
        for y in 0..h {
            for x in 0..w {
                if x > 0 {
                    out.push(' ');
                }
                out.push_str(&fmt_sample(data, y * stride + x * bpp, fmt));
            }
            out.push('\n');
        }
        out.into_bytes()
    }

    /// Serialise as comma-separated values.
    ///
    /// Writes one line per image row, the band-0 sample of each pixel joined by
    /// commas. CSV is a one-band numeric format; the inverse is [`Raster::csv_load`].
    pub fn csv_save(&self) -> Vec<u8> {
        let w = self.width() as usize;
        let h = self.height() as usize;
        let fmt = self.format();
        let bpp = fmt.bytes_per_pixel();
        let stride = self.stride();
        let data = self.data();

        let mut out = String::new();
        for y in 0..h {
            for x in 0..w {
                if x > 0 {
                    out.push(',');
                }
                out.push_str(&fmt_sample(data, y * stride + x * bpp, fmt));
            }
            out.push('\n');
        }
        out.into_bytes()
    }

    /// Encode as a binary Netpbm image (`P5` for one band, `P6` for three).
    ///
    /// 8-bit rasters use a `maxval` of `255`; 16-bit rasters use `65535` and
    /// write each sample most-significant-byte-first, as the Netpbm
    /// specification requires. The inverse is [`Raster::ppm_load`].
    ///
    /// # Errors
    ///
    /// Returns [`EncodeError::Unsupported`] for a float raster (Netpbm has no
    /// integer form for it; the float `PFM` variant is out of scope) and
    /// [`EncodeError::InvalidParameter`] for a band count other than 1 or 3.
    pub fn encode_ppm(&self) -> Result<Vec<u8>, EncodeError> {
        let fmt = self.format();
        if fmt.is_float() {
            return Err(EncodeError::unsupported(
                "ppm (float raster; the PFM float variant is not implemented)",
            ));
        }
        let channels = fmt.channels();
        let magic = match channels {
            1 => "P5",
            3 => "P6",
            other => {
                return Err(EncodeError::InvalidParameter(format!(
                    "Netpbm PPM/PGM support 1 or 3 bands, got {other}"
                )));
            }
        };
        // The two kinds Netpbm has a binary form for. The float guard
        // above has already refused `F32`, and a kind with no Netpbm
        // maxval is refused here rather than written with somebody else's
        // ceiling (issue #607).
        let maxval: u32 = match fmt.kind() {
            SampleKind::U8 => 255,
            SampleKind::U16 => 65535,
            other => {
                return Err(EncodeError::unsupported(format!(
                    "ppm ({other:?} samples have no Netpbm binary form)"
                )));
            }
        };
        let w = self.width();
        let h = self.height();
        let data = self.data();

        let mut out = Vec::with_capacity(32 + data.len());
        out.extend_from_slice(format!("{magic}\n{w} {h}\n{maxval}\n").as_bytes());
        if fmt.kind() == SampleKind::U8 {
            // 8-bit samples are already the interleaved raster body.
            out.extend_from_slice(data);
        } else {
            // 16-bit: native-endian samples out as big-endian.
            for &chunk in data.as_chunks::<2>().0 {
                let v = u16::from_ne_bytes(chunk);
                out.extend_from_slice(&v.to_be_bytes());
            }
        }
        Ok(out)
    }

    /// The shared save routes' entry point: encode the Netpbm container the
    /// **suffix** names, or refuse.
    ///
    /// This is the one row in either route where the suffix picks a container
    /// rather than only a codec. Measured on the pinned vips 8.18.6,
    /// `ppmsave` registers five suffixes and writes something different for
    /// each: `.ppm` a `P6`, `.pgm` a `P5`, `.pbm` a `P4` and `.pfm` a `PF`,
    /// converting the colourspace to whatever the suffix means, while `.pnm`
    /// is refused outright for every interpretation it was handed.
    ///
    /// [`Raster::encode_ppm`] picks its magic from the band count instead, so
    /// the two agree only when the raster already matches the suffix. Where
    /// they disagree this refuses rather than converting, the same call the
    /// `.hdr` row makes (#880): no row in the save table converts, and these
    /// are not going to be the first. The alternative is writing a `P5` body
    /// into a file called `.ppm`, which is the one outcome neither vips nor
    /// Netpbm reads as correct.
    ///
    /// `.pbm` and `.pfm` are not routed here at all, because this build has no
    /// `P4` or `PF` encoder to route them to (issue #882).
    ///
    /// # Errors
    ///
    /// [`EncodeError::Unsupported`] for a suffix that is not `ppm` or `pgm`,
    /// [`EncodeError::InvalidParameter`] for a raster whose band count is not
    /// the one the suffix names, and whatever [`Raster::encode_ppm`] reports
    /// for a sample kind Netpbm has no binary form for.
    pub(crate) fn encode_netpbm(&self, suffix: &str) -> Result<Vec<u8>, EncodeError> {
        let (want, magic) = match suffix {
            "ppm" => (3usize, "P6"),
            "pgm" => (1usize, "P5"),
            other => return Err(EncodeError::unsupported(other.to_owned())),
        };
        let got = self.format().channels();
        if got != want {
            return Err(EncodeError::InvalidParameter(format!(
                ".{suffix} is the {magic} Netpbm container, which carries {want} \
                 bands, and this raster has {got}; vips converts the colourspace \
                 to suit the suffix and libviprs does not"
            )));
        }
        self.encode_ppm()
    }

    /// Serialise as a binary Netpbm image, or empty bytes when unsupported.
    ///
    /// The infallible convenience over [`Raster::encode_ppm`]: it returns the
    /// encoded bytes for the 1- and 3-band `uchar`/`ushort` rasters Netpbm can
    /// carry, and an empty `Vec` for the inputs `encode_ppm` rejects (a float
    /// raster or an unusual band count), mirroring the silent-empty behaviour
    /// libvips itself exhibits when a `ppmsave` target cannot be written.
    pub fn ppm_save(&self) -> Vec<u8> {
        self.encode_ppm().unwrap_or_default()
    }
}

impl Raster {
    /// Decode the libvips `matrix` text format into a single-band float raster.
    ///
    /// Parses the `width height` header (any trailing `scale`/`offset` on that
    /// line is tolerated and ignored) and the `width * height`
    /// whitespace-separated values that follow. The inverse is
    /// [`Raster::matrix_save`].
    ///
    /// # Errors
    ///
    /// Returns a typed [`DecodeError`] when the input is not UTF-8, the header
    /// is missing or non-numeric, a value fails to parse, the value count does
    /// not match the declared dimensions, or the raster cannot be constructed.
    pub fn matrix_load(data: &[u8]) -> Result<Raster, DecodeError> {
        let text =
            std::str::from_utf8(data).map_err(|_| malformed("matrix: input is not valid UTF-8"))?;
        let mut lines = text.lines();
        let header = lines
            .next()
            .ok_or_else(|| malformed("matrix: empty input"))?;
        let mut hdr = header.split_whitespace();
        let width: u32 = hdr
            .next()
            .and_then(|t| t.parse().ok())
            .ok_or_else(|| malformed("matrix: missing or non-numeric width in header"))?;
        let height: u32 = hdr
            .next()
            .and_then(|t| t.parse().ok())
            .ok_or_else(|| malformed("matrix: missing or non-numeric height in header"))?;

        let mut values: Vec<f32> = Vec::new();
        for line in lines {
            for tok in line.split_whitespace() {
                let v: f32 = tok
                    .parse()
                    .map_err(|_| malformed(format!("matrix: non-numeric value {tok:?}")))?;
                values.push(v);
            }
        }

        let expected = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| malformed("matrix: declared dimensions overflow"))?;
        if values.len() != expected {
            return Err(malformed(format!(
                "matrix: expected {expected} values for {width}x{height}, got {}",
                values.len()
            )));
        }
        Ok(Raster::from_f32_samples(width, height, float1()?, &values)?)
    }

    /// Decode a comma-separated-values grid into a single-band float raster.
    ///
    /// The first non-empty row's column count fixes the width; the number of
    /// non-empty rows fixes the height. Blank lines are skipped and fields are
    /// trimmed before parsing.
    ///
    /// Ragged input is tolerated by default, matching libvips' default-lenient
    /// `csvload`: a row shorter than the established width is right-padded with
    /// `0`, and a row longer than it is truncated to the width. (A future
    /// `fail_on` strictness level is where a caller opts into rejecting ragged
    /// rows.) The inverse is [`Raster::csv_save`].
    ///
    /// # Errors
    ///
    /// Returns a typed [`DecodeError`] when the input is not UTF-8, holds no
    /// data rows, contains a non-numeric field, or the raster cannot be
    /// constructed.
    pub fn csv_load(data: &[u8]) -> Result<Raster, DecodeError> {
        let text =
            std::str::from_utf8(data).map_err(|_| malformed("csv: input is not valid UTF-8"))?;
        let mut rows: Vec<Vec<f32>> = Vec::new();
        for line in text.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let mut row: Vec<f32> = Vec::new();
            for field in line.split(',') {
                let t = field.trim();
                let v: f32 = t
                    .parse()
                    .map_err(|_| malformed(format!("csv: non-numeric field {t:?}")))?;
                row.push(v);
            }
            rows.push(row);
        }

        let height = rows.len();
        if height == 0 {
            return Err(malformed("csv: no data rows"));
        }
        // The first row fixes the width; pad short rows with 0 and truncate
        // long ones, so a ragged grid loads rather than erroring (libvips'
        // default-tolerant `csvload`).
        let width = rows[0].len();
        for row in &mut rows {
            row.resize(width, 0.0);
        }
        let width = u32::try_from(width).map_err(|_| malformed("csv: width too large"))?;
        let height = u32::try_from(height).map_err(|_| malformed("csv: height too large"))?;
        let values: Vec<f32> = rows.into_iter().flatten().collect();
        Ok(Raster::from_f32_samples(width, height, float1()?, &values)?)
    }

    /// Decode a Netpbm image (`P1`, `P2`, `P3`, `P4`, `P5`, or `P6`).
    ///
    /// 8-bit (`maxval <= 255`) images decode to `Gray8`/`Rgb8`; 16-bit images
    /// (`maxval` up to `65535`) to `Gray16`/`Rgb16`, with binary samples read
    /// most-significant-byte-first. `#` comments are honoured in the header.
    ///
    /// The bitmap forms `P1` and `P4` carry no `maxval` field and decode to
    /// `Gray8` with **`1` black and `0` white**, which is inverted against
    /// every other format here (issue #919, and `bitmap_ink` carries the
    /// measurement). `P4` pads each row out to a whole byte, so the tail bits
    /// of a short row are padding and not pixels. Nothing in this crate
    /// writes either form; the module docs say why.
    ///
    /// The pixel buffer is bounded before it is reserved: the declared
    /// geometry is priced against the allocation budget through
    /// `DecodeLimits::check_image_alloc` (named rather than linked: it is
    /// `pub(crate)` and these docs are public),
    /// the binary body must be wholly present before any reservation, and the
    /// reservation itself is fallible, so a hostile header cannot force a
    /// multi-gigabyte allocation or a process abort. The inverse is
    /// [`Raster::ppm_save`].
    ///
    /// This runs at [`DecodeLimits::default`](crate::source::DecodeLimits),
    /// which is what the sniffed route uses, so decoding the same bytes by
    /// name and by route gives the same answer (issue #563). It used to cap at
    /// `DEFAULT_MAX_ALLOC_BYTES` instead, **8 GiB** against the route
    /// default's **512 MiB**, so this entry point accepted declared sizes every
    /// other container refused. `decode_netpbm` takes the
    /// limits explicitly (issue #910). Named rather than linked because it is
    /// `pub(crate)` and this method's docs are public, which
    /// `-D rustdoc::private_intra_doc_links` refuses.
    ///
    /// # Errors
    ///
    /// Returns a typed [`DecodeError`] for a bad magic number, a missing or
    /// out-of-range header field, an ASCII sample above the declared `maxval`,
    /// a non-numeric ASCII sample, a truncated binary body, declared
    /// dimensions past the allocation budget, or a raster the dimensions
    /// cannot construct.
    ///
    /// A truncated `P4` body is refused, the same as a truncated `P5` or `P6`
    /// one. vips is not consistent here and this follows the half of it that
    /// is defensible: measured on 8.18.6, a short `P5` body errors on the
    /// first pixel read (`memory area too small`) while a short `P4` body
    /// loads and fills the rows that are not in the file with **black**, so a
    /// `2x2` carrying one of its two body bytes comes back four black pixels
    /// when the byte present says the first row is `white black`. Inventing
    /// pixels is worse than refusing bytes.
    pub fn ppm_load(data: &[u8]) -> Result<Raster, DecodeError> {
        decode_netpbm(data, crate::source::DecodeLimits::default())
    }
}

/// The route table's Netpbm entry point, under the caller's
/// [`DecodeLimits`](crate::source::DecodeLimits).
///
/// [`Raster::ppm_load`] is this with the default limits, so the named entry
/// point and the sniffed one give the same answer for the same bytes, which
/// is the property issue #563 is about.
///
/// # It had a budget, and not the route's budget
///
/// Before issue #910 this capped the declared size against
/// [`DEFAULT_MAX_ALLOC_BYTES`], **8 GiB**, where every route's default is
/// `DecodeLimits::default().max_alloc_bytes`, **512 MiB**. Sixteen times
/// apart, so a declared 4 GiB Netpbm was refused by every other container in
/// the table and accepted here. A component enforcing *a* limit is not the
/// same as one enforcing *the* limit, and a refusal table cannot tell the two
/// apart from outside: the rows pass either way, which is worse than an
/// absent row because it reads as coverage.
///
/// So the ceiling is now [`DecodeLimits::check_image_alloc`], the same call
/// every other native container makes, and it reports
/// [`SourceError::AllocLimitExceeded`] naming the caller's number rather than
/// a constant.
///
/// # Errors
///
/// [`SourceError::AllocLimitExceeded`] if the declared geometry prices past
/// the caller's allocation budget, and the malformed-input errors
/// [`Raster::ppm_load`] documents.
///
/// # The bitmap forms price the raster, not the body
///
/// A `P4` body is one eighth of the raster it builds, rounded up per row, so
/// the two numbers are not interchangeable and only one of them is the
/// allocation. The budget check runs on `width * height * 1 * 1`, the buffer
/// this returns, before the packed body is even located; a check on the packed
/// size would let a caller through at eight times its own ceiling.
pub(crate) fn decode_netpbm(
    data: &[u8],
    limits: crate::source::DecodeLimits,
) -> Result<Raster, DecodeError> {
    {
        let mut pos = 0usize;
        let magic = next_token(data, &mut pos).ok_or_else(|| malformed("ppm: empty input"))?;
        let (channels, ascii, bitmap) = match magic.as_slice() {
            b"P1" => (1usize, true, true),
            b"P2" => (1usize, true, false),
            b"P3" => (3usize, true, false),
            b"P4" => (1usize, false, true),
            b"P5" => (1usize, false, false),
            b"P6" => (3usize, false, false),
            other => {
                let shown = String::from_utf8_lossy(other).into_owned();
                return Err(malformed(format!(
                    "ppm: unrecognised magic number {shown:?}"
                )));
            }
        };

        let width = next_u32(data, &mut pos, "width")?;
        let height = next_u32(data, &mut pos, "height")?;
        // The bitmap forms carry **no `maxval` field**: a `P1`/`P4` sample is
        // one bit, so its range is fixed and the header stops at the height.
        // Reading one anyway would eat the first pixel, which is exactly what
        // vips does to a file that wrongly carries one. Measured on 8.18.6:
        // `P1 / 2 2 / 255 / 0 1 1 0` comes back `0 255 0 0`, so the `255` was
        // consumed as a sample and the last `0` fell off the end.
        let maxval = if bitmap {
            1
        } else {
            next_u32(data, &mut pos, "maxval")?
        };
        if maxval == 0 || maxval > 65535 {
            return Err(malformed(format!(
                "ppm: maxval {maxval} out of range 1..=65535"
            )));
        }
        // The maxval names the kind, not just a width: Netpbm's binary
        // form is unsigned, so a one-byte maxval is `u8` and a two-byte one
        // is `u16` (issue #607).
        let kind = if maxval <= 255 {
            SampleKind::U8
        } else {
            SampleKind::U16
        };
        let bpc = kind.bytes();
        let fmt = PixelFormat::with_kind(channels, kind)
            .ok_or_else(|| malformed("ppm: unsupported channel/kind combination"))?;

        let count = (width as usize)
            .checked_mul(height as usize)
            .and_then(|n| n.checked_mul(channels))
            .ok_or_else(|| malformed("ppm: declared dimensions overflow"))?;
        let need = count
            .checked_mul(bpc)
            .ok_or_else(|| malformed("ppm: declared dimensions overflow"))?;
        // Cap the declared geometry against the **caller's** budget before
        // reserving, so a ~20-byte hostile header cannot request gigabytes,
        // and so this route refuses at the same number every other container
        // refuses at (issue #910). `check_image_alloc` is that shared call.
        limits.check_image_alloc("ppm pixel data", width, height, channels as u64, bpc as u64)?;

        let mut buf: Vec<u8> = Vec::new();
        if ascii {
            // `need` is within budget; reserve fallibly so an in-budget request
            // the host still cannot honour is a typed error, not an abort.
            buf.try_reserve_exact(need)
                .map_err(|_| malformed("ppm: cannot allocate pixel buffer"))?;
            for _ in 0..count {
                let v = next_u32(data, &mut pos, "sample")?;
                if bitmap {
                    // No `maxval` to compare against, and vips does not treat a
                    // sample above 1 as an error: measured on 8.18.6, a `P1`
                    // reading `0 2 7 255` comes back `255 0 0 0`, so every
                    // non-zero value is ink. Refusing them here would make
                    // files vips reads unreachable.
                    buf.push(bitmap_ink(v != 0));
                    continue;
                }
                if v > maxval {
                    return Err(malformed("ppm: sample exceeds maxval"));
                }
                if bpc == 1 {
                    buf.push(v as u8);
                } else {
                    buf.extend_from_slice(&(v as u16).to_ne_bytes());
                }
            }
        } else if bitmap {
            // Exactly one whitespace byte separates the height from the raster.
            if pos < data.len() && data[pos].is_ascii_whitespace() {
                pos += 1;
            }
            // A `P4` row is packed eight pixels to a byte, most significant bit
            // first, and **each row is padded out to a whole byte**, so the body
            // is `ceil(width / 8) * height` rather than `width * height / 8`
            // and the tail bits of a short row are not pixels. Measured on
            // 8.18.6: a 9-wide `P4` whose two body bytes are `80 80` reads as
            // one black pixel, seven white, one black, which only works if the
            // ninth pixel comes from the top bit of the second byte.
            let row_bytes = (width as usize).div_ceil(8);
            let packed = row_bytes
                .checked_mul(height as usize)
                .ok_or_else(|| malformed("ppm: declared dimensions overflow"))?;
            // The packed body is what has to be present, and the *unpacked*
            // raster is what gets reserved: `check_image_alloc` above already
            // priced `width * height`, which is up to eight times the bytes on
            // disk, so the budget bounds what this builds and not what it reads.
            let end = pos
                .checked_add(packed)
                .ok_or_else(|| malformed("ppm: declared dimensions overflow"))?;
            let body = data
                .get(pos..end)
                .ok_or_else(|| malformed("ppm: truncated binary pixel data"))?;
            buf.try_reserve_exact(need)
                .map_err(|_| malformed("ppm: cannot allocate pixel buffer"))?;
            for y in 0..height as usize {
                let row = &body[y * row_bytes..][..row_bytes];
                for x in 0..width as usize {
                    buf.push(bitmap_ink(row[x / 8] & (0x80u8 >> (x % 8)) != 0));
                }
            }
        } else {
            // Exactly one whitespace byte separates the maxval from the raster.
            if pos < data.len() && data[pos].is_ascii_whitespace() {
                pos += 1;
            }
            // Confirm the whole body is present BEFORE reserving, so the
            // reservation is bounded by bytes that actually exist in the input.
            let end = pos
                .checked_add(need)
                .ok_or_else(|| malformed("ppm: declared dimensions overflow"))?;
            let body = data
                .get(pos..end)
                .ok_or_else(|| malformed("ppm: truncated binary pixel data"))?;
            buf.try_reserve_exact(need)
                .map_err(|_| malformed("ppm: cannot allocate pixel buffer"))?;
            if bpc == 1 {
                buf.extend_from_slice(body);
            } else {
                // Binary 16-bit samples are big-endian; store native-endian.
                for &chunk in body.as_chunks::<2>().0 {
                    let v = u16::from_be_bytes(chunk);
                    buf.extend_from_slice(&v.to_ne_bytes());
                }
            }
        }

        Ok(Raster::new(width, height, fmt, buf)?)
    }
}

/// The 8-bit sample a Netpbm bitmap bit decodes to: **set is black**.
///
/// `P1` and `P4` invert the convention every other format in this crate uses.
/// A `1` bit is ink and reads back as `0`; a `0` bit is paper and reads back as
/// `255`. Measured on the pinned vips 8.18.6 rather than taken from the spec:
///
/// ```text
/// $ printf 'P1\n2 2\n0 1 1 0\n' > ascii.pbm
/// $ for y in 0 1; do for x in 0 1; do vips getpoint ascii.pbm $x $y; done; done
/// 255
/// 0
/// 0
/// 255
/// ```
///
/// The source tokens were `0 1 1 0` and the pixels came back `255 0 0 255`.
///
/// It is one function rather than an inline `if` in each of the two decoders
/// because getting it backwards produces a **negative image of the right size,
/// the right band count and the right sample kind**, which no dimension, band
/// or allocation check can see (issue #919). One place to be wrong is one place
/// to test, and `netpbm_bitmap_ink_is_set_is_black` drives it directly.
const fn bitmap_ink(set: bool) -> u8 {
    if set { 0 } else { 255 }
}

/// Read the next whitespace-delimited token from a Netpbm header, skipping
/// leading whitespace and `#` comments (comments run to end of line).
///
/// Returns the token bytes, or `None` at end of input. Used only while reading
/// the ASCII header and ASCII sample stream, never over the binary blob.
fn next_token(data: &[u8], pos: &mut usize) -> Option<Vec<u8>> {
    loop {
        while *pos < data.len() && data[*pos].is_ascii_whitespace() {
            *pos += 1;
        }
        if *pos < data.len() && data[*pos] == b'#' {
            while *pos < data.len() && data[*pos] != b'\n' {
                *pos += 1;
            }
        } else {
            break;
        }
    }
    if *pos >= data.len() {
        return None;
    }
    let start = *pos;
    while *pos < data.len() && !data[*pos].is_ascii_whitespace() && data[*pos] != b'#' {
        *pos += 1;
    }
    Some(data[start..*pos].to_vec())
}

/// Parse the next header token as a `u32`.
fn next_u32(data: &[u8], pos: &mut usize, what: &str) -> Result<u32, DecodeError> {
    let tok = next_token(data, pos).ok_or_else(|| malformed(format!("ppm: missing {what}")))?;
    let s = std::str::from_utf8(&tok).map_err(|_| malformed(format!("ppm: non-ASCII {what}")))?;
    s.parse::<u32>()
        .map_err(|_| malformed(format!("ppm: non-numeric {what} {s:?}")))
}

#[cfg(test)]
mod tests {
    /**
     * Tests that the sniffer claims exactly the six Netpbm magics this
     * crate decodes, and no more (issues #910, #919).
     *
     * The route table's rule is that every row in `SniffedFormat::ALL` can
     * reach a decoder, which `every_container_is_reachable_from_its_own_magic`
     * enforces. So the sniffed set has to be the decodable set: `P1`, `P2` and
     * `P3` for the ASCII forms, `P4`, `P5` and `P6` for the binary ones.
     *
     * **The absences are the assertion, not an oversight.** `PF` is the float
     * PFM and `P7` is PAM, and this crate reads neither by any route, so
     * claiming their magic would put a row in the table that cannot decode.
     *
     * #910 wrote this cell with `P1` and `P4` in the unclaimed half and said
     * in its own body that the assertion was the thing that would go red the
     * day #919 landed. It did, and this is that update rather than a
     * deletion: the two magics moved across, and the half that keeps the rule
     * honest kept its other two rows.
     *
     * An unasserted absence is a coincidence; this one is a rule.
     */
    #[test]
    fn the_sniffer_claims_exactly_the_netpbm_magics_this_crate_decodes() {
        use crate::source::{SniffedFormat, sniff};

        for (magic, body) in [
            (&b"P1"[..], &b"\n2 2\n0 1 1 0\n"[..]),
            (&b"P2"[..], &b"\n2 2\n255\n0 1 2 3\n"[..]),
            (&b"P3"[..], &b"\n1 1\n255\n0 1 2\n"[..]),
            (&b"P4"[..], &b"\n2 2\n\x40\x80"[..]),
            (&b"P5"[..], &b"\n2 2\n255\n\x00\x40\x80\xff"[..]),
            (&b"P6"[..], &b"\n1 1\n255\n\x00\x40\x80"[..]),
        ] {
            let mut bytes = magic.to_vec();
            bytes.extend_from_slice(body);
            let name = String::from_utf8_lossy(magic).into_owned();
            assert_eq!(
                sniff(&bytes),
                Some(SniffedFormat::Netpbm),
                "{name} must be claimed"
            );
            // And claimed *because* it decodes, which is the table's rule.
            assert!(
                decode_netpbm(&bytes, crate::source::DecodeLimits::default()).is_ok(),
                "{name} must decode, or it must not be a row"
            );
        }

        for (magic, body) in [
            (&b"PF"[..], &b"\n2 2\n-1.0\n"[..]),
            (&b"P7"[..], &b"\nWIDTH 2\n"[..]),
        ] {
            let mut bytes = magic.to_vec();
            bytes.extend_from_slice(body);
            let name = String::from_utf8_lossy(magic).into_owned();
            assert_eq!(
                sniff(&bytes),
                None,
                "{name} has no decoder here, so it must not be a row"
            );
            // And unclaimed *because* it does not decode, which is the same
            // rule read the other way. Without this half, deleting the
            // decoder arms would leave the absences passing for the wrong
            // reason.
            assert!(
                decode_netpbm(&bytes, crate::source::DecodeLimits::default()).is_err(),
                "{name} must not decode, or it must be a row"
            );
        }
    }

    /**
     * Tests that the Netpbm route refuses at the **caller's** budget and not
     * at a constant of its own (issue #910).
     *
     * This is the whole substance of the change and it is invisible from a
     * refusal table. Before #910 this path capped against
     * `DEFAULT_MAX_ALLOC_BYTES`, **8 GiB**, where every route's default is
     * `DecodeLimits::default().max_alloc_bytes`, **512 MiB**. Sixteen times
     * apart, so a declared 4 GiB Netpbm was refused by every other container
     * and accepted here. It had *a* budget and not *the* budget, and a row in
     * `tests/decode_alloc_refusal_shape.rs` would have passed either way,
     * which is worse than an absent row because it reads as coverage.
     *
     * Only varying the caller's limit can see the difference, so that is what
     * this does: the same 3 MiB header is accepted under a 512 MiB budget and
     * refused under a 1 MiB one, and the refusal names **1 MiB**. A route
     * still using the constant would accept both and a route hard-coding any
     * other number would report that number.
     */
    #[test]
    fn the_netpbm_route_refuses_at_the_callers_budget_not_at_one_of_its_own() {
        // 1024 x 1024 RGB8 is 3 MiB: comfortably inside the 512 MiB default
        // and comfortably outside a 1 MiB ceiling.
        let header = b"P6\n1024 1024\n255\n";
        let mut bytes = header.to_vec();
        bytes.resize(header.len() + 1024 * 1024 * 3, 0);

        // Positive control: it decodes at all, so the refusal below is the
        // ceiling and not the parser.
        let ok = decode_netpbm(&bytes, crate::source::DecodeLimits::default())
            .expect("3 MiB is inside the 512 MiB default");
        assert_eq!((ok.width(), ok.height()), (1024, 1024));

        let tight = crate::source::DecodeLimits::default().with_max_alloc_bytes(1024 * 1024);
        let err = decode_netpbm(&bytes, tight).expect_err("3 MiB is outside a 1 MiB ceiling");
        match &err {
            crate::source::SourceError::AllocLimitExceeded {
                needed_bytes,
                max_alloc_bytes,
                ..
            } => {
                assert_eq!(
                    *max_alloc_bytes,
                    1024 * 1024,
                    "the refusal must name the caller's ceiling, not a constant"
                );
                assert_eq!(*needed_bytes, 1024 * 1024 * 3);
            }
            other => panic!("expected the route's alloc refusal, got {other:?}"),
        }
    }

    /**
     * Tests [`Raster::encode_netpbm`]'s refusal arm directly, because neither
     * route can reach it (issue #882).
     *
     * Both callers match `"ppm" | "pgm"` before calling, so the `other` arm is
     * unreachable through `Raster::save` and `Raster::encode_to_buffer` alike,
     * and a mutation that replaced it with a silent `P6` fallback left the
     * whole suite green. Every route check still passed, because every route
     * check only ever asks for a suffix the routes already matched.
     *
     * That is the shape #696's first bullet is about, arriving through a
     * different door: a branch the tests cannot drive is a branch nothing
     * holds. So this drives it directly, with the two rows the routes do have
     * as the positive control that the function works at all.
     */
    #[test]
    fn encode_netpbm_refuses_a_suffix_it_has_no_container_for() {
        let rgb = Raster::new(2, 2, PixelFormat::Rgb8, (0..12u8).collect()).unwrap();
        for suffix in ["pbm", "pfm", "pnm", "ppm2", ""] {
            let err = rgb
                .encode_netpbm(suffix)
                .expect_err("this build has no container for .{suffix}");
            assert!(
                matches!(&err, EncodeError::Unsupported { format } if format == suffix),
                ".{suffix} must be refused by name, got {err}"
            );
        }
        // Positive control: the two it does have.
        assert!(rgb.encode_netpbm("ppm").unwrap().starts_with(b"P6"));
        let gray = Raster::new(2, 2, PixelFormat::Gray8, vec![0, 64, 128, 255]).unwrap();
        assert!(gray.encode_netpbm("pgm").unwrap().starts_with(b"P5"));
    }

    use super::*;

    fn float1_test() -> PixelFormat {
        PixelFormat::with_kind(1, SampleKind::F32).expect("FloatF32(1) is a valid format")
    }

    /// A small single-band float raster whose values are all exactly
    /// representable in `f32`, so any lossless codec round-trips it bit-for-bit.
    fn synth_float() -> Raster {
        let samples = [0.0f32, 1.0, -2.5, 0.25, 100.0, -0.75];
        Raster::from_f32_samples(3, 2, float1_test(), &samples).expect("well-formed float raster")
    }

    fn max_abs_diff(a: &Raster, b: &Raster) -> f32 {
        let sa = a.f32_samples().expect("float raster");
        let sb = b.f32_samples().expect("float raster");
        sa.iter()
            .zip(&sb)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn matrix_round_trips_losslessly() {
        let r = synth_float();
        let bytes = r.matrix_save();
        let back = Raster::matrix_load(&bytes).expect("matrix reload");
        assert_eq!((back.width(), back.height()), (r.width(), r.height()));
        assert_eq!(max_abs_diff(&r, &back), 0.0);
    }

    #[test]
    fn matrix_header_ignores_optional_scale_offset() {
        // libvips admits a "width height scale offset" header; the extra
        // trailing numbers must not be mistaken for pixel values.
        let body = b"2 1 1 0\n3.5 -4.25\n";
        let r = Raster::matrix_load(body).expect("matrix reload");
        assert_eq!((r.width(), r.height()), (2, 1));
        assert_eq!(r.f32_samples().expect("float"), vec![3.5, -4.25]);
    }

    #[test]
    fn csv_round_trips_losslessly() {
        let r = synth_float();
        let bytes = r.csv_save();
        let back = Raster::csv_load(&bytes).expect("csv reload");
        assert_eq!((back.width(), back.height()), (r.width(), r.height()));
        assert_eq!(max_abs_diff(&r, &back), 0.0);
    }

    #[test]
    fn ppm_binary_p5_p6_round_trip() {
        let gray = Raster::new(2, 2, PixelFormat::Gray8, vec![0, 64, 128, 255])
            .expect("well-formed gray raster");
        let p5 = gray.ppm_save();
        assert!(p5.starts_with(b"P5"), "1-band saves as PGM/P5");
        let gray_back = Raster::ppm_load(&p5).expect("p5 reload");
        assert_eq!(gray_back.format(), PixelFormat::Gray8);
        assert_eq!(gray_back.data(), gray.data());

        let rgb = Raster::new(2, 1, PixelFormat::Rgb8, vec![1, 2, 3, 4, 5, 6])
            .expect("well-formed rgb raster");
        let p6 = rgb.ppm_save();
        assert!(p6.starts_with(b"P6"), "3-band saves as PPM/P6");
        let rgb_back = Raster::ppm_load(&p6).expect("p6 reload");
        assert_eq!(rgb_back.format(), PixelFormat::Rgb8);
        assert_eq!(rgb_back.data(), rgb.data());

        // The fallible encode variant produces the exact same bytes.
        assert_eq!(rgb.encode_ppm().expect("encode_ppm"), p6);
    }

    #[test]
    fn ppm_ascii_p2_p3_load() {
        let p2 = b"P2\n2 2\n255\n0 64\n128 255\n";
        let gray = Raster::ppm_load(p2).expect("p2 reload");
        assert_eq!(gray.format(), PixelFormat::Gray8);
        assert_eq!((gray.width(), gray.height()), (2, 2));
        assert_eq!(gray.data(), &[0u8, 64, 128, 255]);

        // P3 with a header comment, which the reader must skip.
        let p3 = b"P3\n# a comment line\n2 1\n255\n1 2 3 4 5 6\n";
        let rgb = Raster::ppm_load(p3).expect("p3 reload");
        assert_eq!(rgb.format(), PixelFormat::Rgb8);
        assert_eq!(rgb.data(), &[1u8, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn ppm_16bit_round_trips_big_endian() {
        let samples: Vec<u8> = [1000u16, 2000, 60000, 65535]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        let gray = Raster::new(2, 2, PixelFormat::Gray16, samples).expect("gray16 raster");
        let p5 = gray.ppm_save();
        assert!(
            p5.windows(5).any(|w| w == b"65535"),
            "16-bit maxval in header"
        );
        let back = Raster::ppm_load(&p5).expect("16-bit p5 reload");
        assert_eq!(back.format(), PixelFormat::Gray16);
        assert_eq!(back.data(), gray.data());
    }

    #[test]
    fn matrix_load_rejects_malformed_without_panic() {
        let err =
            Raster::matrix_load(b"not a matrix header\nfoo bar").expect_err("must be typed error");
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn csv_load_rejects_non_numeric_without_panic() {
        let err = Raster::csv_load(b"1,2,3\n4,oops,6\n").expect_err("must be typed error");
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn ppm_load_rejects_bad_magic_without_panic() {
        let err = Raster::ppm_load(b"PZ\n1 1\n255\n\x00").expect_err("must be typed error");
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn ppm_load_rejects_truncated_binary_without_panic() {
        // Header promises 4 gray samples but only 2 bytes of body follow.
        let err = Raster::ppm_load(b"P5\n2 2\n255\n\x01\x02").expect_err("must be typed error");
        assert!(!err.to_string().is_empty());
    }

    #[test]
    fn encode_ppm_rejects_float_raster() {
        let r = synth_float();
        let err = r
            .encode_ppm()
            .expect_err("float has no integer Netpbm form");
        assert!(matches!(err, EncodeError::Unsupported { .. }));
        // The infallible convenience returns empty bytes for the unsupported case.
        assert!(r.ppm_save().is_empty());
    }

    #[test]
    fn loaders_resolve_as_raster_associated_functions() {
        // The acceptance cells call the decoders as inherent associated
        // functions on `Raster` (`Raster::matrix_load(&bytes)`), not as free
        // functions. Drive each one through that path so the surface the cells
        // depend on cannot silently regress to a free fn.
        let r = synth_float();
        let m = <Raster>::matrix_load(&r.matrix_save()).expect("matrix reload");
        assert_eq!((m.width(), m.height()), (r.width(), r.height()));
        let c = <Raster>::csv_load(&r.csv_save()).expect("csv reload");
        assert_eq!((c.width(), c.height()), (r.width(), r.height()));
        let gray =
            Raster::new(2, 1, PixelFormat::Gray8, vec![7, 8]).expect("well-formed gray raster");
        let p = <Raster>::ppm_load(&gray.ppm_save()).expect("ppm reload");
        assert_eq!(p.data(), gray.data());
    }

    #[test]
    fn csv_load_pads_ragged_rows_by_default() {
        // libvips' `csvload` is default-tolerant: a short row is padded with 0
        // rather than rejected. This is the surface `test_fail_on` pins (a
        // ragged CSV must load OK by default).
        let r = Raster::csv_load(b"1,2,3\n4,5").expect("ragged csv loads by default");
        assert_eq!((r.width(), r.height()), (3, 2));
        assert_eq!(
            r.f32_samples().expect("float"),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 0.0]
        );

        // A row longer than the first is truncated to the established width.
        let wide = Raster::csv_load(b"1,2\n3,4,5,6").expect("over-long row truncates");
        assert_eq!((wide.width(), wide.height()), (2, 2));
        assert_eq!(wide.f32_samples().expect("float"), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn ppm_load_rejects_oversized_header_dimensions() {
        // A ~20-byte hostile binary header declares billions of pixels. The
        // body-length check must fire before any reservation, so this is a
        // typed Err, never a multi-gigabyte allocation or a process abort.
        let hostile = b"P5\n65535 65535\n65535\n";
        let err = Raster::ppm_load(hostile).expect_err("oversized header must be a typed error");
        assert!(!err.to_string().is_empty());

        // Dimensions whose byte size exceeds the allocation budget are rejected
        // up front on the ASCII path too, before any per-sample parsing. The
        // variant is the route's, `AllocLimitExceeded`, and not the raster
        // constructor's `ByteBudgetExceeded`, because this path now prices
        // against the caller's `DecodeLimits` like every other container
        // rather than against `DEFAULT_MAX_ALLOC_BYTES` (issue #910).
        let over_budget = b"P3\n65535 65535\n255\n";
        let err =
            Raster::ppm_load(over_budget).expect_err("over-budget header must be a typed error");
        assert!(
            matches!(err, DecodeError::AllocLimitExceeded { .. }),
            "expected the route's alloc refusal, got {err:?}"
        );
    }

    #[test]
    fn ppm_ascii_rejects_sample_above_maxval() {
        // An ASCII sample greater than the declared maxval is malformed input.
        let err = Raster::ppm_load(b"P2\n1 1\n10\n99\n").expect_err("sample above maxval");
        assert!(err.to_string().contains("maxval"));
    }

    /**
     * Tests that this module dispatches on sample kind and never on byte
     * width, by asserting that neither the byte-width accessor on
     * [`PixelFormat`] nor its width-keyed constructor survives in
     * `src/textio.rs`.
     * Works by scanning the module's own source, compiled in with
     * `include_str!`, for the accessor's name; the needle is spelled in two
     * halves so this assertion is not itself a hit. A byte width is not a
     * sample kind: four bytes is `f32` today and would be `u32` under issue
     * #517, so the sites this replaced would print a 32-bit integer sample as a float (issue #607).
     * Input: `src/textio.rs` -> Output: zero occurrences.
     */
    #[test]
    fn textio_does_not_dispatch_on_byte_width() {
        const SRC: &str = include_str!("textio.rs");
        let needles = [
            concat!("bytes_per_", "channel"),
            concat!("with_", "channels"),
        ];
        // Positive control: the same scan over the same string finds a token
        // that is present, so the zero below is a real zero and not the
        // vacuous pass an empty read would give.
        assert!(
            SRC.contains(concat!("fn ", "fmt_sample")),
            "positive control failed: the scan cannot see this module's source"
        );
        for needle in needles {
            assert_eq!(
                SRC.matches(needle).count(),
                0,
                "{needle} is back in src/textio.rs; dispatch on \
                 PixelFormat::kind() and PixelFormat::with_kind() instead"
            );
        }
    }

    /**
     * Tests the bitmap polarity directly, on the one function that carries it
     * (issue #919).
     *
     * `P1` and `P4` invert the convention every other format here uses: a set
     * bit is ink and decodes to `0`, a clear bit is paper and decodes to
     * `255`. Getting it backwards produces a negative image of the right
     * width, the right height, the right band count and the right sample
     * kind, so no dimension check, band check or allocation check can see it.
     * That is why the polarity is one `const fn` and why it has a cell of its
     * own rather than only being covered through the two decoders: a direct
     * cell on the helper fails on the mutation, where an end-to-end cell can
     * be argued about.
     *
     * Both directions, because "everything is white" and "everything is
     * black" each pass half of this.
     * Input: `bitmap_ink(true)` and `bitmap_ink(false)` -> Output: `0` and
     * `255`.
     */
    #[test]
    fn netpbm_bitmap_ink_is_set_is_black() {
        assert_eq!(bitmap_ink(true), 0, "a set bit is ink, so it decodes to 0");
        assert_eq!(
            bitmap_ink(false),
            255,
            "a clear bit is paper, so it decodes to 255"
        );
    }

    /**
     * Tests that a `P1` ASCII bitmap decodes with `0` as white, pinned to the
     * four pixels measured on the pinned vips 8.18.6 (issue #919).
     *
     * ```text
     * $ printf 'P1\n2 2\n0 1 1 0\n' > ascii.pbm
     * $ vipsheader ascii.pbm
     * ascii.pbm: 2x2 uchar, 1 band, b-w, ppmload
     * $ for y in 0 1; do for x in 0 1; do vips getpoint ascii.pbm $x $y; done; done
     * 255
     * 0
     * 0
     * 255
     * ```
     *
     * The tokens were `0 1 1 0` and the pixels came back `255 0 0 255`. The
     * fixture is deliberately not symmetric under inversion, so a reader that
     * gets the polarity backwards cannot pass this.
     * Input: `P1 / 2 2 / 0 1 1 0` -> Output: a 2x2 `Gray8` raster holding
     * `255 0 0 255`.
     */
    #[test]
    fn a_p1_ascii_bitmap_decodes_with_zero_as_white() {
        let raster = Raster::ppm_load(b"P1\n2 2\n0 1 1 0\n").expect("P1 is a decodable form");
        assert_eq!((raster.width(), raster.height()), (2, 2));
        assert_eq!(raster.format(), PixelFormat::Gray8);
        assert_eq!(
            raster.data(),
            &[255, 0, 0, 255],
            "0 is white and 1 is black, which is inverted against every other \
             format here (issue #919)"
        );
    }

    /**
     * Tests that a `P1` sample above `1` is still ink rather than a refusal,
     * and that a `P1` header has no `maxval` field to compare it against
     * (issue #919).
     *
     * Two measurements on the pinned 8.18.6, and the second is the one that
     * proves the header shape:
     *
     * ```text
     * $ printf 'P1\n2 2\n0 2 7 255\n'     -> 255 0 0 0
     * $ printf 'P1\n2 2\n255\n0 1 1 0\n'  -> 0 255 0 0
     * ```
     *
     * The first says every non-zero value is ink, so refusing `2` would make
     * files vips reads unreachable. The second says the `255` was **consumed
     * as a pixel**: a reader that took it for a `maxval` would answer
     * `255 0 0 255` instead, so this fixture separates "there is no maxval
     * field" from "there is one and it happens not to matter".
     * Input: the two files above -> Output: `255 0 0 0` and `0 255 0 0`.
     */
    #[test]
    fn a_p1_sample_above_one_is_ink_and_the_header_carries_no_maxval() {
        let above_one = Raster::ppm_load(b"P1\n2 2\n0 2 7 255\n").expect("P1 accepts 2, 7, 255");
        assert_eq!(
            above_one.data(),
            &[255, 0, 0, 0],
            "every non-zero sample is ink, measured"
        );

        let stray_maxval =
            Raster::ppm_load(b"P1\n2 2\n255\n0 1 1 0\n").expect("the stray 255 is a pixel");
        assert_eq!(
            stray_maxval.data(),
            &[0, 255, 0, 0],
            "a P1 header stops at the height, so the 255 is the first pixel \
             and the trailing 0 falls off the end"
        );
    }

    /**
     * Tests that a `P4` binary bitmap unpacks most significant bit first and
     * that **each row** is padded out to a whole byte (issue #919).
     *
     * Nine pixels wide on purpose: eight would be a round number where a
     * dropped `div_ceil` is the identity, and nine puts the last pixel of
     * each row alone in a second byte with seven padding bits behind it.
     * Measured on the pinned 8.18.6:
     *
     * ```text
     * $ printf 'P4\n9 2\n\x80\x80\x01\x00' > bin.pbm
     * row 0 -> 0 255 255 255 255 255 255 255 0
     * row 1 -> 255 255 255 255 255 255 255 0 255
     * ```
     *
     * Row 0's bytes are `80 80`: the first pixel is the top bit of byte one
     * and the ninth is the top bit of byte two, which only works if the row
     * restarts on a byte boundary. Row 1's `01 00` puts its only ink in the
     * *bottom* bit of its first byte, the eighth pixel, so a reader that
     * unpacked least significant bit first would put it first instead.
     * Input: the file above -> Output: the two rows as measured.
     */
    #[test]
    fn a_p4_binary_bitmap_unpacks_msb_first_and_pads_each_row_to_a_byte() {
        let raster =
            Raster::ppm_load(b"P4\n9 2\n\x80\x80\x01\x00").expect("P4 is a decodable form");
        assert_eq!((raster.width(), raster.height()), (9, 2));
        assert_eq!(raster.format(), PixelFormat::Gray8);
        assert_eq!(
            raster.data(),
            &[
                0, 255, 255, 255, 255, 255, 255, 255, 0, //
                255, 255, 255, 255, 255, 255, 255, 0, 255,
            ],
            "nine pixels a row, two bytes a row, most significant bit first"
        );
    }

    /**
     * Tests that the bits past the end of a `P4` row are padding and not
     * pixels (issue #919).
     *
     * The row-padding rule is only load-bearing if the padding is *ignored*,
     * and a decoder that reads `width * height` bits as one continuous stream
     * passes the previous cell's first row and fails here. So the fixture
     * fills every padding bit with the opposite of what it should be:
     * measured on 8.18.6, `P4 / 2 2 / 7f bf` reads `255 0 / 0 255` even
     * though `7f` is six set bits after its two pixels and `bf` is six set
     * bits after its two.
     * Input: `P4 / 2 2 / 7f bf` -> Output: `255 0 0 255`, the same picture as
     * the two-byte fixture whose padding is all zero.
     */
    #[test]
    fn a_p4_rows_padding_bits_are_not_pixels() {
        let junk_padding = Raster::ppm_load(b"P4\n2 2\n\x7f\xbf").expect("P4 with junk padding");
        assert_eq!(
            junk_padding.data(),
            &[255, 0, 0, 255],
            "six padding bits a row, none of them pixels"
        );

        // The same picture written with clean padding, as the control that
        // says the assertion above is about the padding and not about the
        // fixture.
        let clean_padding = Raster::ppm_load(b"P4\n2 2\n\x40\x80").expect("P4 with clean padding");
        assert_eq!(junk_padding.data(), clean_padding.data());
    }

    /**
     * Tests that a truncated `P4` body is refused rather than filled in
     * (issue #919).
     *
     * vips is not consistent here and this follows the half of it that is
     * defensible. Measured on 8.18.6: a short `P5` body errors on the first
     * pixel read (`VipsImage: memory area too small`), while a short `P4`
     * body loads and fills the rows that are not in the file with **black**,
     * so `P4 / 2 2 / 40` comes back four black pixels when the one byte
     * present says the first row is `white black`. Inventing pixels is worse
     * than refusing bytes, and this crate already refuses a short `P5`.
     *
     * The full two-byte file beside it is the positive control, so "refused"
     * cannot pass because `P4` is broken generally.
     * Input: `P4 / 2 2 / 40` and `P4 / 2 2 / 40 80` -> Output: a truncation
     * error, and `255 0 0 255`.
     */
    #[test]
    fn a_truncated_p4_body_is_refused_where_vips_fills_it_with_black() {
        let err = Raster::ppm_load(b"P4\n2 2\n\x40").expect_err("one byte is half a 2x2 bitmap");
        assert!(
            err.to_string().contains("truncated"),
            "a short P4 body must be named as truncated, got {err}"
        );

        let whole = Raster::ppm_load(b"P4\n2 2\n\x40\x80").expect("both body bytes present");
        assert_eq!(whole.data(), &[255, 0, 0, 255]);
    }

    /**
     * Tests that the `P4` route prices the raster it builds and not the body
     * it reads, at the **caller's** ceiling (issue #919).
     *
     * A `P4` body is one eighth of the raster it unpacks into, rounded up per
     * row, so the two numbers are not interchangeable and only one of them is
     * the allocation. `1033 x 1031` unpacks to 1 065 023 bytes from a body of
     * 134 030, which is a factor of about eight: under a 512 KiB ceiling the
     * raster is over and the body is comfortably under, so a check placed on
     * the packed size would let this through at eight times the caller's own
     * limit and a refusal table could not tell the difference.
     *
     * The refusal has to name both numbers for that to be an assertion rather
     * than a coincidence, so it asserts `max_alloc_bytes` is the caller's
     * 512 KiB (not a constant of the route's own, which is what #910 fixed)
     * and `needed_bytes` is the unpacked 1 065 023 (not the packed 134 030).
     * The default-limits decode beside it is the positive control that the
     * bytes are a valid bitmap.
     *
     * The width is not a multiple of eight and neither dimension is round,
     * because a round one is where a dropped `div_ceil` is the identity.
     * Input: a 1033x1031 `P4` under a 512 KiB budget and under the default ->
     * Output: `AllocLimitExceeded { needed_bytes: 1065023, max_alloc_bytes:
     * 524288 }`, and a decoded raster.
     */
    #[test]
    fn the_p4_route_prices_the_raster_it_builds_not_the_body_it_reads() {
        const WIDTH: usize = 1033;
        const HEIGHT: usize = 1031;
        const ROW_BYTES: usize = WIDTH.div_ceil(8); // 130
        let header = format!("P4\n{WIDTH} {HEIGHT}\n");
        let mut bytes = header.into_bytes();
        bytes.resize(bytes.len() + ROW_BYTES * HEIGHT, 0);

        // Positive control: the bytes are a valid bitmap, so the refusal
        // below is the ceiling and not the parser.
        let ok = decode_netpbm(&bytes, crate::source::DecodeLimits::default())
            .expect("1 MiB is inside the 512 MiB default");
        assert_eq!((ok.width() as usize, ok.height() as usize), (WIDTH, HEIGHT));

        let tight = crate::source::DecodeLimits::default().with_max_alloc_bytes(512 * 1024);
        let err = decode_netpbm(&bytes, tight).expect_err("the unpacked raster is over 512 KiB");
        match &err {
            crate::source::SourceError::AllocLimitExceeded {
                needed_bytes,
                max_alloc_bytes,
                ..
            } => {
                assert_eq!(
                    *max_alloc_bytes,
                    512 * 1024,
                    "the refusal must name the caller's ceiling, not a constant"
                );
                assert_eq!(
                    *needed_bytes,
                    (WIDTH * HEIGHT) as u64,
                    "the price is the unpacked raster, not the {} packed body bytes",
                    ROW_BYTES * HEIGHT
                );
            }
            other => panic!("expected the route's alloc refusal, got {other:?}"),
        }
    }

    /**
     * Tests that the two bitmap forms decode and that nothing in this crate
     * writes either, which is the decision #919 made rather than an omission
     * (issue #919).
     *
     * The property #882 established, and #910 restored, is that everything
     * this crate **writes** it can **read**. It has never been the converse:
     * `P2` and `P3` have decoded since #77 and no route has ever written
     * either, so the decode set has always been strictly wider than the
     * encode set. Adding `P1` and `P4` widens it by two more and leaves the
     * invariant that matters untouched.
     *
     * Writing `P4` would need a thresholding policy and there is no oracle to
     * copy: measured on 8.18.6, `vips ppmsave` to `.pbm` right-aligns the
     * last partial byte of every row where its own reader takes those bits
     * from the top, so a 5-wide bitmap it writes reads back as a different
     * picture. The module docs carry the four widths. Matching those bytes
     * means writing files the oracle misreads and writing correct bytes means
     * diverging from it, so neither is parity, which is the same ground #882
     * stood on for `.pbm` and `.pfm`.
     *
     * This is the pin, so the day an encoder does arrive the asymmetry goes
     * red and says so rather than being a surprise. The two decodes are its
     * positive control, so "nothing writes them" cannot pass because nothing
     * works.
     * Input: a `P1` and a `P4` file, `encode_netpbm("pbm")`, and
     * `encode_ppm` on a bilevel `Gray8` raster -> Output: both decode, `.pbm`
     * is refused by name, and the encoder still writes `P5`.
     */
    #[test]
    fn the_bitmap_forms_decode_and_nothing_here_writes_them() {
        let p1 = Raster::ppm_load(b"P1\n2 2\n0 1 1 0\n").expect("P1 decodes");
        let p4 = Raster::ppm_load(b"P4\n2 2\n\x40\x80").expect("P4 decodes");
        assert_eq!(p1.data(), &[255, 0, 0, 255]);
        assert_eq!(p4.data(), p1.data(), "the same picture in both forms");

        // `.pbm` is the suffix `ppmsave` writes a `P4` to, and it is still
        // refused by name here.
        let err = p1
            .encode_netpbm("pbm")
            .expect_err(".pbm has no encoder here");
        assert!(
            matches!(&err, EncodeError::Unsupported { format } if format == "pbm"),
            ".pbm must be refused by name, got {err}"
        );

        // And a raster that *is* bilevel still encodes as `P5`, so no writer
        // can quietly start choosing the bitmap form without this saying so.
        let written = p1.encode_ppm().expect("a Gray8 raster encodes");
        assert!(
            written.starts_with(b"P5"),
            "a one-band raster encodes as P5, whatever its values"
        );

        // The invariant that actually matters, on the same bytes: what this
        // crate writes, it reads.
        let back = Raster::ppm_load(&written).expect("what we write we read");
        assert_eq!(back.data(), p1.data());
    }

    /**
     * Tests that a bitmap decodes the same by name and by route, which is the
     * property issue #563 is about (issues #563, #919).
     *
     * `Raster::ppm_load` and the sniffed route are two entry points onto one
     * decoder, and a magic wired into `SniffedFormat::Netpbm`'s row without a
     * matching arm in `decode_netpbm` would sniff and then fail. Both forms,
     * because the ASCII and binary halves are separate code paths and a round
     * trip through one says nothing about the other.
     * Input: a `P1` and a `P4` file through `Raster::ppm_load` and
     * `crate::decode_bytes` -> Output: the same raster from both, four times.
     */
    #[test]
    fn a_bitmap_decodes_the_same_by_name_and_by_route() {
        for bytes in [&b"P1\n2 2\n0 1 1 0\n"[..], &b"P4\n2 2\n\x40\x80"[..]] {
            let named = Raster::ppm_load(bytes).expect("the named entry point decodes it");
            let routed = crate::decode_bytes(bytes).expect("the sniffed route decodes it");
            assert_eq!(routed.format(), named.format());
            assert_eq!(
                (routed.width(), routed.height()),
                (named.width(), named.height())
            );
            assert_eq!(routed.data(), named.data());
            assert_eq!(named.data(), &[255, 0, 0, 255]);
        }
    }
}
