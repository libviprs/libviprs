//! Radiance HDR (`.hdr`) load and save: RGBE bytes in, three-band float out.
//!
//! Ported from libvips `foreign/radiance.c` (the container, itself
//! copy-pasted from Greg Ward's Radiance 5.4 sources) together with
//! `colour/rad2float.c` and `colour/float2rad.c` (the sample codec), which
//! libvips keeps in separate files because it models Radiance as a
//! *coding*, a 4-band uchar raster tagged `VIPS_CODING_RAD` that any real
//! operation silently unpacks. libviprs has no coding concept, so this
//! module fuses the two halves: [`decode_radiance`] is `radload` composed
//! with `rad2float`, and [`Raster::encode_radiance`] is `float2rad`
//! composed with `radsave`.
//!
//! # Operations
//!
//! | libviprs method | libvips equivalent | result |
//! |---|---|---|
//! | [`decode_radiance`] | `radload` + `rad2float` | [`PixelFormat::FloatF32`]`(3)` raster, always tagged [`Interpretation::ScRgb`] |
//! | [`Raster::encode_radiance`] | `float2rad` + `radsave_buffer` | `.hdr` bytes |
//! | [`Raster::save_radiance`] | `float2rad` + `radsave` | `.hdr` file |
//!
//! # Semantics
//!
//! * **The carrier is float, never RGBE.** vips decodes to a 4-band uchar
//!   raster tagged `VIPS_CODING_RAD` (`radiance.c:706-709`) and unpacks it
//!   to 3-band float scRGB the moment any operation touches it. libviprs
//!   decodes straight to `FloatF32(3)`. The 4-band-`Rgba8` alternative is
//!   disqualified on correctness: `resample.rs` premultiplies on
//!   `format.has_alpha()` and `resize.rs` forks its downscale kernel on it,
//!   so a 4-band RGBE raster tagged `Rgba8` would be premultiplied by its
//!   own exponent byte. `Interpretation` is a tag and nothing in
//!   `resize.rs` consults it.
//! * **Accepted divergence.** `vipsheader` on a libviprs-loaded `.hdr`
//!   reports `bands 3 / float / coding none` where vips reports
//!   `bands 4 / uchar / coding rad`. The RGBE carrier is observable in vips
//!   only through `vipsheader` and `copy`. [`crate::encode_tiff`] makes the
//!   same trade.
//! * **Decode is half-bit centred.** `rad2float.c`'s `colr_color()` is
//!   `(mantissa + 0.5) * 2^(e - 136)`, where `136` is `COLXS + 8` with
//!   `COLXS` 128, and an exponent byte of 0 is a hard zero rather than
//!   `0.5 * 2^-136`. The `image` crate uses the plain
//!   `mantissa * 2^(e - 136)`, a 100% error at mantissa 0, which is why
//!   this module hand-rolls its codec and why `image`'s `hdr` feature stays
//!   off in `Cargo.toml`.
//! * **Encode is the matched half.** `float2rad.c`'s `setcolr()` is
//!   `frexp(max) * 255.9999 / max` with a `1e-32` floor, negatives clamped
//!   to zero, and a *truncating* conversion to `u8`. The two constants only
//!   make sense together; see [`decode_radiance`] for the exact fixed-point
//!   domain this module pins.
//! * **The size range picks an encoding, it does not gate.** `MINELEN` 8
//!   and `MAXELEN` 0x7fff select run-length encoding; outside `8..=32767`
//!   `scanline_write` (`radiance.c:955-978`) writes flat, unencoded
//!   scanlines. Measured on vips 8.18.4: width 4 and width 40000 both save
//!   successfully, with 4-bytes-per-pixel flat payloads.
//! * **Hardened RLE.** Both of vips's guards are ported: the
//!   `rshift > 24` bail-out on chained old-style repeat markers
//!   (`radiance.c:392-397`) and the scanline-length check
//!   (`radiance.c:437-440`), plus the run overrun check
//!   (`radiance.c:451-454`). See [`RadianceError`].
//! * **The `FORMAT=` line is ignored on load, on purpose.** vips picks the
//!   colour tag from it at `radiance.c:693-698`, but that arm is
//!   unreachable: `radiance.c:636` calls `formatval(line, read->format)`
//!   while `radiance.c:314` declares `formatval(char fmt[MAXFMTLEN], const
//!   char *s)` with `fmt` as the output buffer, so the arguments are
//!   swapped and `read->format` keeps its `COLRFMT` default. Measured, a
//!   file declaring `FORMAT=32-bit_rle_xyze` still reports
//!   `rad-format: 32-bit_rle_rgbe` and `interpretation: scrgb`. libviprs
//!   reproduces that rather than the evident intent, because the tag
//!   reaches pixels through `colourspace()` and an `Xyz` tag would break
//!   op-surface parity. **If upstream fixes `formatval`, libviprs should
//!   follow.** The save side is unaffected and still writes
//!   `32-bit_rle_xyze` for an `Xyz` raster (`radiance.c:899-901`, which
//!   reads `in->Type` and is live).
//! * **`.hdr` only.** `vips -l` registers exactly one suffix
//!   (`vips__rad_suffs`, `radiance.c:1035`) and the magic is exactly the
//!   first line `#?RADIANCE` (`vips__rad_israd`, `radiance.c:568-577`).
//!   `#?RGBE` is rejected; measured, such a file falls through to
//!   `magickload`.
//! * **A loaded `.hdr` cannot enter the pyramid engine.** See
//!   [`decode_radiance`].
//!
//! Every number this module is pinned against was measured on the real
//! vips 8.18.4 binary and is recorded, with the commands that produced it,
//! in `oracle-captures/foreign-radiance/`.
//!
//! Every entry point here is fallible and there is no panicking twin,
//! matching the rest of the codec surface in [`crate::encode`],
//! [`crate::webp`], and [`crate::gif`]: a decoder's failures come from
//! untrusted bytes, so a panicking spelling would have no honest caller.

use std::path::Path;

use thiserror::Error;

use crate::codec::EncodeError;
use crate::conversion::Interpretation;
use crate::imageio::{MetadataValue, SaveError};
use crate::pixel::PixelFormat;
use crate::raster::{Raster, RasterError, decode_alloc_bytes};
use crate::source::{DecodeLimits, SourceError};

/// The exact magic line every Radiance file opens with
/// (`vips__rad_israd`, `radiance.c:574`).
pub(crate) const MAGIC: &[u8] = b"#?RADIANCE";

/// The `FORMAT=` value for an RGB file (`COLRFMT`, `radiance.c:202`).
const FORMAT_RGBE: &str = "32-bit_rle_rgbe";

/// The `FORMAT=` value for a CIE XYZ file (`CIEFMT`, `radiance.c:203`).
const FORMAT_XYZE: &str = "32-bit_rle_xyze";

/// Excess added to the stored exponent byte (`COLXS`, `radiance.c:180`).
const COLXS: i32 = 128;

/// Total exponent bias applied on decode: `COLXS + 8`, the `8` folding in
/// the mantissa's own scale (`rad2float.c`, `colr_color()`).
const EXP_BIAS: i32 = COLXS + 8;

/// Shortest scanline the run-length encoding is used for
/// (`MINELEN`, `radiance.c:235`).
const MINELEN: u32 = 8;

/// Longest scanline the run-length encoding is used for
/// (`MAXELEN`, `radiance.c:236`).
const MAXELEN: u32 = 0x7fff;

/// Shortest repeat the encoder emits as a run rather than as literals
/// (`MINRUN`, `radiance.c:237`).
const MINRUN: usize = 4;

/// Mantissa floor below which `setcolr` writes an all-zero pixel
/// (`float2rad.c`).
const ZERO_FLOOR: f64 = 1e-32;

/// Mantissa scale `setcolr` multiplies the `frexp` significand by
/// (`float2rad.c`).
const MANTISSA_SCALE: f64 = 255.9999;

/// Bands in the decoded raster, and the only band count
/// [`Raster::encode_radiance`] accepts.
const BANDS: usize = 3;

/// Longest header line accepted before the file is called malformed.
///
/// libvips reads header lines into a fixed 4096-byte buffer and silently
/// skips whatever does not fit (`vips_sbuf_get_line`, `sbuf.c`). libviprs
/// borrows from the input instead of copying, so the cap is here only to
/// stop a header with no newline in it from being scanned end to end.
const MAX_HEADER_LINE: usize = 4096;

/// Errors from the Radiance codec.
///
/// Every variant except [`RadianceError::Raster`] describes a specific
/// malformation in untrusted bytes, which is what makes them worth typing:
/// the fuzz corpus in `fuzz/corpus/fuzz_radiance/` asserts on the variant,
/// not on a message.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RadianceError {
    /// The first line is not exactly `#?RADIANCE`.
    ///
    /// vips checks the same way and rejects the near-miss `#?RGBE`
    /// (`vips__rad_israd`, `radiance.c:568-577`).
    #[error("radiance: expected the magic line \"#?RADIANCE\", found {found:?}")]
    BadMagic {
        /// The first line as read, lossily decoded and truncated for the
        /// message.
        found: String,
    },
    /// The file ends inside the header, before the blank line that
    /// terminates it or before the resolution line that follows
    /// (`getheader`, `radiance.c:334-352`).
    #[error("radiance: the file ends inside the header, before {expected}")]
    TruncatedHeader {
        /// What the parser was still waiting for.
        expected: &'static str,
    },
    /// The resolution line does not carry both an `X` and a `Y` axis with
    /// positive extents (`str2resolu`, `radiance.c:276-305`).
    #[error("radiance: unparseable resolution line {line:?}")]
    BadResolution {
        /// The line as read, lossily decoded.
        line: String,
    },
    /// The declared geometry is zero, negative, or beyond the
    /// `VIPS_MAX_COORD` ceiling vips applies at `radiance.c:698-704`.
    #[error("radiance: declared image size {width}x{height} is out of bounds")]
    DimensionOutOfBounds {
        /// The declared scanline length.
        width: i64,
        /// The declared number of scanlines.
        height: i64,
    },
    /// Decoding the declared geometry would allocate more than
    /// [`DecodeLimits::max_alloc_bytes`].
    ///
    /// A `.hdr` body is run-length encoded, so a tiny file can declare a
    /// very large image; this is the budget that bounds it.
    #[error(
        "radiance: decoding {width}x{height} needs {needed} bytes, above the \
         {max_alloc_bytes}-byte decode allocation budget"
    )]
    AllocLimitExceeded {
        /// The declared scanline length.
        width: u32,
        /// The declared number of scanlines.
        height: u32,
        /// Bytes the decoded raster would occupy.
        needed: u64,
        /// The budget from [`DecodeLimits::max_alloc_bytes`].
        max_alloc_bytes: u64,
    },
    /// A run-length-encoded scanline's own length marker disagrees with the
    /// width the header declared (`radiance.c:437-440`).
    ///
    /// The `image` crate never makes this comparison, which is why it is
    /// the cheapest available defence against a desynchronised stream.
    #[error("radiance: scanline {row} declares length {declared} but the image is {width} wide")]
    ScanlineLengthMismatch {
        /// The scanline the mismatch was found on, from the top.
        row: u32,
        /// The length the scanline's marker declared.
        declared: u32,
        /// The width the header declared.
        width: u32,
    },
    /// A run or literal block would write past the end of its scanline
    /// (`radiance.c:451-454`).
    #[error("radiance: an encoded run overruns scanline {row}")]
    ScanlineOverrun {
        /// The scanline the overrun was found on, from the top.
        row: u32,
    },
    /// More than four consecutive old-style repeat markers were chained.
    ///
    /// Each marker shifts its count eight bits further left, so vips bails
    /// once the shift would pass 24 (`radiance.c:392-397`). Without that
    /// guard the count grows without bound; that is exactly the `image`
    /// crate overflow tracked as #539.
    #[error("radiance: scanline {row} chains more than four consecutive old-style repeat markers")]
    RunawayRepeat {
        /// The scanline the chain was found on, from the top.
        row: u32,
    },
    /// An old-style repeat marker appeared with no preceding pixel to
    /// repeat.
    ///
    /// vips reads one pixel before the start of its scanline buffer here
    /// (`copycolr(scanline[0], scanline[-1])`, `radiance.c:387`); libviprs
    /// rejects the file instead.
    #[error("radiance: scanline {row} opens with a repeat marker, which has no pixel to repeat")]
    RepeatWithoutPixel {
        /// The scanline the marker was found on, from the top.
        row: u32,
    },
    /// The pixel data ends before every declared scanline has been read.
    #[error("radiance: the file ends inside scanline {row}")]
    TruncatedScanline {
        /// The scanline the file ended inside, from the top.
        row: u32,
    },
    /// A header line ran past the line cap without a newline, so the file
    /// is not a header libviprs is willing to scan.
    ///
    /// libvips reads header lines into a fixed 4096-byte buffer and
    /// silently skips whatever does not fit (`vips_sbuf_get_line`,
    /// `sbuf.c`); libviprs borrows from the input instead of copying, so
    /// the cap exists only to stop a header with no newline in it from
    /// being scanned end to end.
    #[error("radiance: a header line runs past the {cap}-byte cap without a newline")]
    HeaderLineTooLong {
        /// The cap the line exceeded, in bytes.
        cap: usize,
    },
    /// A `COLORCORR=` or `PRIMARIES=` line did not carry the three or eight
    /// numbers it promises. libvips fails the whole load here too
    /// (`rad2vips_process_line`, `radiance.c:632-660`).
    #[error("radiance: malformed header line {line:?}")]
    BadHeaderLine {
        /// The line as read, lossily decoded.
        line: String,
    },
    /// Constructing the decoded [`Raster`] failed.
    #[error(transparent)]
    Raster(#[from] RasterError),
}

/// Options for [`Raster::encode_radiance`] and [`Raster::save_radiance`]
/// (libvips `radsave` / `radsave_buffer`).
///
/// Plain, `Default`, and module-scoped, so callers write
/// `radiance::SaveOptions { exposure: Some(2.0), ..Default::default() }`
/// and later fields can be added without a breaking change.
///
/// Both fields are `None` by default, meaning "take the value from the
/// raster's own `rad-expos` / `rad-aspect` field, and fall back to `1.0`".
/// That is what `vips2rad_make_header` (`radiance.c:876-919`) does, so a
/// `.hdr` that libviprs loaded and saved keeps the header scalars it came
/// with. `Some(v)` overrides both.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SaveOptions {
    /// The `EXPOSURE=` header value.
    pub exposure: Option<f64>,
    /// The `PIXASPECT=` header value.
    pub aspect: Option<f64>,
}

/// Decode Radiance `.hdr` bytes into a three-band float [`Raster`]
/// (libvips `radload` followed by `rad2float`).
///
/// The result is [`PixelFormat::FloatF32`]`(3)`, always tagged
/// [`Interpretation::ScRgb`]. Samples are linear radiance values with no
/// upper bound, which is the point of the format: a bright pixel reads
/// back as `4088.0`, not as a clipped `1.0`.
///
/// The `FORMAT=` header line is read past and ignored, and so is
/// `32-bit_rle_xyze` in particular, because vips ignores it too: the tag
/// arm at `radiance.c:693-698` is unreachable behind an argument-order
/// defect at `radiance.c:636`. Matching the reference is deliberate here
/// rather than inherited, because the interpretation tag is consumed by
/// [`Raster::colourspace`] and a different tag would move pixels, not just
/// the header. See the module docs for the full reproduction; if upstream
/// fixes it, libviprs should follow.
///
/// # The fixed point
///
/// `float2rad` after `rad2float` reproduces the original RGBE quadruple
/// exactly when its largest mantissa is at least 128 and its exponent byte
/// is in `23..=255`, verified here over 298,240 combinations. Those are
/// precisely the quadruples an encoder produces, so a `.hdr` written by
/// vips or by [`Raster::encode_radiance`] round-trips byte for byte.
/// Outside that domain the pair renormalises rather than repeating itself:
/// below exponent 23 the `1e-32` floor collapses the pixel to all-zero, and
/// a largest mantissa under 128 is rescaled (`(127, 63, 31, 129)` becomes
/// `(254, 126, 62, 128)`).
///
/// # Limitation: no pyramid
///
/// A `FloatF32(3)` raster is **rejected by the pyramid engine**.
/// [`crate::resize::downscale_half`] and `downscale_to` both return
/// [`RasterError::FloatUnsupported`] for a float format, so a loaded `.hdr`
/// cannot be fed to [`crate::EngineBuilder`], which is this crate's
/// headline feature. The resampling surface in [`crate::resample`] does
/// handle float, so ordinary operations work; only the tiled-pyramid path
/// is closed. Cast to an integer format first if you need a pyramid, and
/// accept that doing so throws the high dynamic range away.
///
/// # Errors
///
/// * [`SourceError::Radiance`] wrapping any [`RadianceError`] variant: a
///   bad magic line, a truncated or unparseable header, out-of-bounds
///   geometry, an over-budget allocation, or any of the four malformed-RLE
///   cases.
/// * [`SourceError::CoordLimitExceeded`] when either declared axis exceeds
///   [`DecodeLimits::max_coord`].
/// * [`SourceError::DimensionLimitExceeded`] when `width * height` exceeds
///   [`DecodeLimits::max_pixels`].
pub fn decode_radiance(bytes: &[u8], limits: DecodeLimits) -> Result<Raster, SourceError> {
    let mut cursor = Cursor::new(bytes);
    let header = read_header(&mut cursor)?;
    let (width, height) = (header.width, header.height);

    limits.check_coord(width, height)?;
    limits.check_pixels(width, height)?;
    // The price and the comparison are the crate's, not this module's
    // (issue #632). This one used to be a plain `*`, reachable only
    // because `parse_resolution` bounds each axis below
    // `DEFAULT_MAX_COORD` before `DecodeLimits` is consulted at all, so
    // the product could not leave a `u64`. That is a different check's
    // guarantee, spelled nowhere near this line, and the codecs that
    // copied the shape do not all have one. The typed variant is retagged
    // from `check_alloc`'s rather than replacing it; #632 deferred
    // collapsing the per-format variants.
    let needed = decode_alloc_bytes(width, height, BANDS as u64, SAMPLE_BYTES as u64);
    limits
        .check_alloc("Radiance pixel buffer", needed)
        .map_err(|_| RadianceError::AllocLimitExceeded {
            width,
            height,
            needed,
            max_alloc_bytes: limits.max_alloc_bytes,
        })?;

    let row_samples = width as usize * BANDS;
    let mut data = vec![0u8; row_samples * height as usize * SAMPLE_BYTES];
    let mut scanline = vec![[0u8; 4]; width as usize];
    for row in 0..height {
        read_scanline(&mut cursor, &mut scanline, row)?;
        let base = row as usize * row_samples * SAMPLE_BYTES;
        for (x, quad) in scanline.iter().enumerate() {
            let rgb = rgbe_to_float(*quad);
            for (band, sample) in rgb.iter().enumerate() {
                let off = base + (x * BANDS + band) * SAMPLE_BYTES;
                data[off..off + SAMPLE_BYTES].copy_from_slice(&sample.to_ne_bytes());
            }
        }
    }

    let mut raster =
        Raster::new(width, height, float_rgb(), data).map_err(RadianceError::Raster)?;
    // Always scRGB, never XYZ. `radiance.c:693-698` picks the tag from the
    // `FORMAT=` line, but the line is never parsed: `rad2vips_process_line`
    // calls `formatval(line, read->format)` at `radiance.c:636` while
    // `radiance.c:314` declares `formatval(char fmt[MAXFMTLEN], const char
    // *s)` with `fmt` as the OUTPUT buffer, so the arguments are swapped and
    // `read->format` keeps the `COLRFMT` default it was given at
    // `radiance.c:610`. The XYZ arm is unreachable in every 8.18.x build.
    // Measured: a file declaring `FORMAT=32-bit_rle_xyze` still reports
    // `rad-format: 32-bit_rle_rgbe` and `interpretation: scrgb`.
    //
    // libviprs matches the reference rather than the evident intent, because
    // this tag reaches PIXELS and not just the header: `colourspace()` reads
    // it, so an `Xyz` tag would make an XYZE file convert to sRGB
    // differently here than in vips. That is an op-surface parity break,
    // which is a worse trade than the header divergence this module already
    // accepts. If upstream fixes `formatval`, libviprs should follow and
    // start honouring `FORMAT=`; the save side already writes
    // `32-bit_rle_xyze` for an `Xyz` raster, so only this line moves.
    raster.meta.interpretation = Some(Interpretation::ScRgb);
    header.attach_fields(&mut raster);
    Ok(raster)
}

impl Raster {
    /// Encode as Radiance `.hdr` bytes (libvips `float2rad` followed by
    /// `radsave_buffer`).
    ///
    /// Only [`PixelFormat::FloatF32`]`(3)` is accepted, and its samples are
    /// written as linear radiance with no colourspace transform.
    ///
    /// # Divergence from `vips radsave`
    ///
    /// `vips radsave` on an image that is not already `VIPS_CODING_RAD`
    /// routes through `vips_colourspace(-> sRGB)` and clips to uchar.
    /// Measured on the reference suite's 141x980 `sample.hdr`:
    /// `rad2float` then `radsave` gives max 254.5 / avg 228.068244 where
    /// the original is max 7728 / avg 51.784629, while `float2rad` then
    /// `radsave` reproduces the original exactly. So in vips the only
    /// HDR-preserving save path is `float2rad` *then* `radsave`, and that
    /// is the pair this method is equivalent to. libviprs round-trips high
    /// dynamic range; a bare `vips radsave` does not.
    ///
    /// # Errors
    ///
    /// [`EncodeError::Encode`] when the raster is not `FloatF32(3)`, or
    /// when its dimensions do not fit the format's 32-bit resolution line.
    pub fn encode_radiance(&self, options: SaveOptions) -> Result<Vec<u8>, EncodeError> {
        let format = self.format();
        if format != float_rgb() {
            return Err(EncodeError::encode(format!(
                "radiance carries three float bands; {format:?} has no RGBE spelling, \
                 cast to FloatF32(3) first"
            )));
        }
        let (width, height) = (self.width(), self.height());
        // The resolution line is `%8d` of a C `int` (`resolu2str`,
        // `radiance.c:263-275`), so anything past `i32::MAX` has no
        // spelling in the format at all.
        if width > i32::MAX as u32 || height > i32::MAX as u32 {
            return Err(EncodeError::encode(format!(
                "radiance cannot spell a {width}x{height} resolution line;                  both axes must fit a signed 32-bit integer"
            )));
        }

        let header = Header::for_save(self, options);
        let mut out = header.write(width, height);

        let mut scanline = vec![[0u8; 4]; width as usize];
        let data = self.data();
        let stride = self.stride();
        for row in 0..height as usize {
            let base = row * stride;
            for (x, quad) in scanline.iter_mut().enumerate() {
                let mut rgb = [0.0f64; BANDS];
                for (band, sample) in rgb.iter_mut().enumerate() {
                    let off = base + (x * BANDS + band) * SAMPLE_BYTES;
                    let mut raw = [0u8; SAMPLE_BYTES];
                    raw.copy_from_slice(&data[off..off + SAMPLE_BYTES]);
                    *sample = f64::from(f32::from_ne_bytes(raw));
                }
                *quad = float_to_rgbe(rgb);
            }
            write_scanline(&mut out, &scanline);
        }
        Ok(out)
    }

    /// Save the raster to `path` as Radiance `.hdr` (libvips `radsave`).
    ///
    /// # Errors
    ///
    /// [`SaveError::Encode`] when [`Raster::encode_radiance`] rejects the
    /// raster, or [`SaveError::Io`] when the file write fails.
    pub fn save_radiance(&self, path: &Path, options: SaveOptions) -> Result<(), SaveError> {
        let bytes = self.encode_radiance(options).map_err(|e| match e {
            EncodeError::Io(io) => SaveError::Io(io),
            other => SaveError::Encode(crate::sink::SinkError::EncodeMsg(other.to_string())),
        })?;
        std::fs::write(path, bytes)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Bytes per float sample.
const SAMPLE_BYTES: usize = 4;

/// The pixel format every decode produces and the only one
/// [`Raster::encode_radiance`] accepts.
fn float_rgb() -> PixelFormat {
    PixelFormat::FloatF32(
        std::num::NonZeroU16::new(BANDS as u16).expect("the band count is non-zero"),
    )
}

/// Whether a scanline of this width is run-length encoded.
///
/// `MINELEN` and `MAXELEN` (`radiance.c:235-236`) pick an encoding; they do
/// not gate the save. Outside the range, `scanline_write`
/// (`radiance.c:955-978`) writes flat, unencoded pixels and
/// `scanline_read` (`radiance.c:404-484`) reads them back the same way.
fn uses_run_length(width: u32) -> bool {
    (MINELEN..=MAXELEN).contains(&width)
}

/// `2^n`, built straight from the exponent field so it is exact for every
/// `n` this module uses (`-135..=119` on decode).
fn exp2(n: i32) -> f64 {
    debug_assert!(
        (-1022..=1023).contains(&n),
        "exp2 is only exact for normals"
    );
    f64::from_bits(((n + 1023) as u64) << 52)
}

/// Split `x` into a significand in `[0.5, 1)` and a power of two, the way
/// C's `frexp` does. `float2rad.c`'s `setcolr` depends on this exactly:
/// substituting `log2().floor() + 1` disagrees with it around powers of two.
fn frexp(x: f64) -> (f64, i32) {
    if x == 0.0 || !x.is_finite() {
        return (x, 0);
    }
    const EXP_MASK: u64 = 0x7ff << 52;
    let (bits, offset) = {
        let raw = x.to_bits();
        if raw & EXP_MASK == 0 {
            // Subnormal. Unreachable from `setcolr`, whose `1e-32` floor
            // sits far above it, but cheap to get right.
            ((x * exp2(64)).to_bits(), -64)
        } else {
            (raw, 0)
        }
    };
    let raw_exp = ((bits >> 52) & 0x7ff) as i32;
    let significand = f64::from_bits((bits & !EXP_MASK) | (1022 << 52));
    (significand, raw_exp - 1022 + offset)
}

/// Decode one RGBE quadruple (`colr_color`, `rad2float.c`).
///
/// The `+ 0.5` is the half-bit centring that separates libvips from the
/// `image` crate, and the `exponent == 0` arm is a hard zero rather than
/// `0.5 * 2^-136`.
fn rgbe_to_float(quad: [u8; 4]) -> [f32; BANDS] {
    if quad[3] == 0 {
        return [0.0; BANDS];
    }
    let scale = exp2(i32::from(quad[3]) - EXP_BIAS);
    [
        ((f64::from(quad[0]) + 0.5) * scale) as f32,
        ((f64::from(quad[1]) + 0.5) * scale) as f32,
        ((f64::from(quad[2]) + 0.5) * scale) as f32,
    ]
}

/// Encode one RGB triple (`setcolr`, `float2rad.c`).
///
/// The conversion to `u8` truncates, as C's does. Where C is undefined
/// (a non-finite product) Rust saturates and maps NaN to zero, which is
/// what the measured macOS build of vips 8.18.4 also produces.
fn float_to_rgbe(rgb: [f64; BANDS]) -> [u8; 4] {
    let mut d = if rgb[0] > rgb[1] { rgb[0] } else { rgb[1] };
    if rgb[2] > d {
        d = rgb[2];
    }
    if d <= ZERO_FLOOR {
        return [0; 4];
    }
    let (significand, exponent) = frexp(d);
    let scale = significand * MANTISSA_SCALE / d;
    let sample = |v: f64| if v > 0.0 { (v * scale) as u8 } else { 0 };
    [
        sample(rgb[0]),
        sample(rgb[1]),
        sample(rgb[2]),
        (exponent + COLXS) as u8,
    ]
}

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------

/// The nominal CRT primaries and EE white point Radiance assumes when a
/// file carries no `PRIMARIES=` line (`radiance.c:191-199`, `read_new`).
///
/// libvips stores these in a `float` array, so the doubles it later
/// publishes are the `f32` roundings; libviprs matches that rather than
/// publishing a value the reference implementation cannot produce.
const DEFAULT_PRIMS: [[f32; 2]; 4] = [
    [0.640, 0.330],
    [0.290, 0.600],
    [0.150, 0.060],
    [1.0 / 3.0, 1.0 / 3.0],
];

/// libvips's field names for the eight primary chromaticities
/// (`prims_name`, `radiance.c:664-669`).
const PRIMS_NAMES: [[&str; 2]; 4] = [
    ["rad-prims-rx", "rad-prims-ry"],
    ["rad-prims-gx", "rad-prims-gy"],
    ["rad-prims-bx", "rad-prims-by"],
    ["rad-prims-wx", "rad-prims-wy"],
];

/// libvips's field names for the three colour-correction factors
/// (`colcor_name`, `radiance.c:671-675`).
const COLCOR_NAMES: [&str; BANDS] = ["rad-colcor-r", "rad-colcor-g", "rad-colcor-b"];

/// Everything the Radiance header carries, in the same shape as libvips's
/// `Read` / `Write` structs (`radiance.c:552-566`, `radiance.c:820-833`).
struct Header {
    format: String,
    expos: f64,
    colcor: [f32; BANDS],
    aspect: f64,
    prims: [[f32; 2]; 4],
    width: u32,
    height: u32,
}

impl Header {
    /// The defaults `read_new` / `write_new` install before any header line
    /// is seen.
    fn defaults() -> Self {
        Self {
            format: FORMAT_RGBE.to_string(),
            expos: 1.0,
            colcor: [1.0; BANDS],
            aspect: 1.0,
            prims: DEFAULT_PRIMS,
            width: 0,
            height: 0,
        }
    }

    /// Publish the header scalars under libvips's own field names, so a
    /// `.hdr` libviprs loaded and saved keeps what it came with.
    fn attach_fields(&self, raster: &mut Raster) {
        raster.fields.set("rad-format", self.format.clone().into());
        raster.fields.set("rad-expos", self.expos.into());
        for (name, value) in COLCOR_NAMES.iter().zip(self.colcor) {
            raster.fields.set(name, f64::from(value).into());
        }
        raster.fields.set("rad-aspect", self.aspect.into());
        for (names, values) in PRIMS_NAMES.iter().zip(self.prims) {
            for (name, value) in names.iter().zip(values) {
                raster.fields.set(name, f64::from(value).into());
            }
        }
    }

    /// Rebuild the header for a save, exactly as `vips2rad_make_header`
    /// does (`radiance.c:876-919`): read each scalar back off the raster
    /// when it carries one, then let the interpretation override the
    /// `FORMAT=` line.
    fn for_save(raster: &Raster, options: SaveOptions) -> Self {
        let mut header = Self::defaults();
        // The typed accessors on `MetadataValue` panic on a mismatched
        // variant, and every one of these fields is caller-writable through
        // `Raster::set_field`, so read them by pattern instead: a field of
        // the wrong shape falls back to the Radiance default rather than
        // taking down a save.
        let double = |name: &str| match raster.get_field(name) {
            Some(MetadataValue::Double(v)) => Some(v),
            Some(MetadataValue::Int(v)) => Some(v as f64),
            _ => None,
        };

        if let Some(MetadataValue::Str(value)) = raster.get_field("rad-format") {
            let value = sanitise_header_value(&value);
            if !value.is_empty() {
                header.format = value;
            }
        }
        header.expos = options
            .exposure
            .or_else(|| double("rad-expos"))
            .unwrap_or(1.0);
        header.aspect = options
            .aspect
            .or_else(|| double("rad-aspect"))
            .unwrap_or(1.0);
        for (slot, name) in header.colcor.iter_mut().zip(COLCOR_NAMES) {
            if let Some(v) = double(name) {
                *slot = v as f32;
            }
        }
        for (slots, names) in header.prims.iter_mut().zip(PRIMS_NAMES) {
            for (slot, name) in slots.iter_mut().zip(names) {
                if let Some(v) = double(name) {
                    *slot = v as f32;
                }
            }
        }
        match raster.interpretation() {
            Interpretation::ScRgb => header.format = FORMAT_RGBE.to_string(),
            Interpretation::Xyz => header.format = FORMAT_XYZE.to_string(),
            _ => {}
        }
        header
    }

    /// Serialise the header and the resolution line, byte for byte as
    /// `vips2rad_put_header` does (`radiance.c:921-947`), except for the
    /// `SOFTWARE=` line: claiming to be vips would be a lie.
    fn write(&self, width: u32, height: u32) -> Vec<u8> {
        let mut out = String::new();
        out.push_str("#?RADIANCE\n");
        out.push_str(&format!("FORMAT={}\n", self.format));
        out.push_str(&format!("EXPOSURE={}\n", c_exponential(self.expos)));
        out.push_str(&format!(
            "COLORCORR= {:.6} {:.6} {:.6}\n",
            self.colcor[0], self.colcor[1], self.colcor[2]
        ));
        out.push_str(&format!(
            "SOFTWARE=libviprs {}\n",
            env!("CARGO_PKG_VERSION")
        ));
        out.push_str(&format!("PIXASPECT={:.6}\n", self.aspect));
        out.push_str("PRIMARIES=");
        for pair in self.prims {
            for value in pair {
                out.push_str(&format!(" {value:.4}"));
            }
        }
        out.push('\n');
        out.push('\n');
        // Always `-Y h +X w`: `vips2rad_make_header` stamps `YDECR | YMAJOR`
        // "for consistency with vips" (`radiance.c:916-919`).
        out.push_str(&format!("-Y {height:>8} +X {width:>8}\n"));
        out.into_bytes()
    }
}

/// Strip anything that could end a header line early.
///
/// `rad-format` is read off the raster, and `Raster::set_field` is public,
/// so without this a caller-supplied newline would inject header lines into
/// a saved file.
fn sanitise_header_value(value: &str) -> String {
    value
        .chars()
        .filter(|c| !c.is_control())
        .collect::<String>()
        .trim()
        .to_string()
}

/// C's `%e`: six fraction digits and an at-least-two-digit signed
/// exponent, which Rust's `{:e}` does not produce.
fn c_exponential(value: f64) -> String {
    let formatted = format!("{value:.6e}");
    let (mantissa, exponent) = formatted
        .split_once('e')
        .expect("Rust's LowerExp always emits an 'e'");
    let exponent: i32 = exponent.parse().expect("LowerExp emits a decimal exponent");
    let sign = if exponent < 0 { '-' } else { '+' };
    format!("{mantissa}e{sign}{:02}", exponent.abs())
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/// A position in the untrusted file bytes. Every read is bounds-checked and
/// returns `None` at the end, so a malformed file becomes a typed error
/// rather than a panic or an over-read.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.pos
    }

    /// One `\n`-delimited line with any trailing `\r` stripped, matching
    /// `vips_sbuf_get_line` (`sbuf.c`), which also treats end-of-input as a
    /// line terminator. `Err` means the line ran past [`MAX_HEADER_LINE`].
    fn line(&mut self) -> Result<Option<&'a [u8]>, RadianceError> {
        if self.pos >= self.data.len() {
            return Ok(None);
        }
        let rest = &self.data[self.pos..];
        let window = rest.len().min(MAX_HEADER_LINE);
        match rest[..window].iter().position(|&b| b == b'\n') {
            Some(end) => {
                self.pos += end + 1;
                Ok(Some(strip_cr(&rest[..end])))
            }
            None if rest.len() <= MAX_HEADER_LINE => {
                self.pos = self.data.len();
                Ok(Some(strip_cr(rest)))
            }
            None => Err(RadianceError::HeaderLineTooLong {
                cap: MAX_HEADER_LINE,
            }),
        }
    }

    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let end = self.pos.checked_add(n)?;
        let slice = self.data.get(self.pos..end)?;
        self.pos = end;
        Some(slice)
    }

    fn quad(&mut self) -> Option<[u8; 4]> {
        let bytes = self.take(4)?;
        Some([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    fn peek(&self) -> Option<u8> {
        self.data.get(self.pos).copied()
    }

    fn byte(&mut self) -> Option<u8> {
        let value = self.peek()?;
        self.pos += 1;
        Some(value)
    }
}

/// Drop a single trailing `\r`, so DOS line endings parse.
fn strip_cr(line: &[u8]) -> &[u8] {
    match line.split_last() {
        Some((b'\r', head)) => head,
        _ => line,
    }
}

/// C's `atof`: the longest numeric prefix, or zero.
fn atof(text: &[u8]) -> f64 {
    let mut i = 0;
    while i < text.len() && text[i].is_ascii_whitespace() {
        i += 1;
    }
    let start = i;
    if i < text.len() && (text[i] == b'+' || text[i] == b'-') {
        i += 1;
    }
    while i < text.len() && text[i].is_ascii_digit() {
        i += 1;
    }
    if i < text.len() && text[i] == b'.' {
        i += 1;
        while i < text.len() && text[i].is_ascii_digit() {
            i += 1;
        }
    }
    if i < text.len() && text[i] | 0x20 == b'e' {
        let mut j = i + 1;
        if j < text.len() && (text[j] == b'+' || text[j] == b'-') {
            j += 1;
        }
        if j < text.len() && text[j].is_ascii_digit() {
            while j < text.len() && text[j].is_ascii_digit() {
                j += 1;
            }
            i = j;
        }
    }
    std::str::from_utf8(&text[start..i])
        .ok()
        .and_then(|t| t.parse::<f64>().ok())
        .unwrap_or(0.0)
}

/// `sscanf("%f %f ...")`: the first `N` whitespace-separated tokens, all of
/// which must parse, extra tokens ignored. `None` when fewer than `N`
/// parse, which is how libvips decides a `COLORCORR=` or `PRIMARIES=` line
/// is malformed.
fn scan_floats<const N: usize>(text: &[u8]) -> Option<[f32; N]> {
    let mut out = [0.0f32; N];
    let mut tokens = text
        .split(|b| b.is_ascii_whitespace())
        .filter(|t| !t.is_empty());
    for slot in &mut out {
        let token = std::str::from_utf8(tokens.next()?).ok()?;
        *slot = token.parse::<f32>().ok()?;
    }
    Some(out)
}

/// Read the header lines, the blank line that ends them, and the resolution
/// line that follows (`getheader` + `str2resolu`, `radiance.c:334-352` and
/// `radiance.c:276-305`).
fn read_header(cursor: &mut Cursor<'_>) -> Result<Header, RadianceError> {
    let first = cursor.line()?.unwrap_or_default();
    if first != MAGIC {
        return Err(RadianceError::BadMagic {
            found: String::from_utf8_lossy(&first[..first.len().min(32)]).into_owned(),
        });
    }

    let mut header = Header::defaults();
    loop {
        let Some(line) = cursor.line()? else {
            return Err(RadianceError::TruncatedHeader {
                expected: "the blank line that ends the header",
            });
        };
        if line.is_empty() {
            break;
        }
        process_header_line(line, &mut header)?;
    }

    let Some(resolution) = cursor.line()? else {
        return Err(RadianceError::TruncatedHeader {
            expected: "the resolution line",
        });
    };
    let (width, height) = parse_resolution(resolution)?;
    // `radiance.c:698-704`: the same bounds vips applies to the declared
    // geometry before it initialises the image.
    let max = i64::from(crate::imageio::DEFAULT_MAX_COORD);
    if width <= 0 || height <= 0 || width >= max || height >= max {
        return Err(RadianceError::DimensionOutOfBounds { width, height });
    }
    header.width = width as u32;
    header.height = height as u32;
    Ok(header)
}

/// One header line (`rad2vips_process_line`, `radiance.c:632-660`).
///
/// The `FORMAT=` arm is deliberately empty, and that is a port of a
/// libvips defect rather than an oversight. `radiance.c:636` calls
/// `formatval(line, read->format)` while `radiance.c:314` declares
/// `formatval(char fmt[MAXFMTLEN], const char *s)` with `fmt` as the
/// **output** buffer: the arguments are swapped, so the call parses the
/// empty destination instead of the header line, matches nothing, and
/// returns 0 without writing. `read->format` therefore keeps the `COLRFMT`
/// default installed at `radiance.c:610`, and the `XYZ` branch at
/// `radiance.c:695-696` is unreachable. Measured on 8.18.4: a file
/// declaring `FORMAT=32-bit_rle_xyze` still reports
/// `rad-format: 32-bit_rle_rgbe` and `interpretation: scrgb`.
///
/// libviprs reproduces that, so `rad-format` always reads back as
/// `32-bit_rle_rgbe` and [`decode_radiance`] always tags
/// [`Interpretation::ScRgb`]. Honouring the line would put a third
/// behaviour in the world, matching neither the source nor the binary, and
/// it would move pixels rather than just the header. If upstream fixes
/// `formatval`, libviprs should follow.
fn process_header_line(line: &[u8], header: &mut Header) -> Result<(), RadianceError> {
    let malformed = || RadianceError::BadHeaderLine {
        line: String::from_utf8_lossy(&line[..line.len().min(80)]).into_owned(),
    };
    if line.starts_with(b"FORMAT=".as_slice()) {
        // Recognised and discarded, exactly as vips does. See above.
    } else if let Some(rest) = line.strip_prefix(b"EXPOSURE=".as_slice()) {
        header.expos *= atof(rest);
    } else if let Some(rest) = line.strip_prefix(b"COLORCORR=".as_slice()) {
        let values = scan_floats::<BANDS>(rest).ok_or_else(malformed)?;
        for (slot, value) in header.colcor.iter_mut().zip(values) {
            *slot *= value;
        }
    } else if let Some(rest) = line.strip_prefix(b"PIXASPECT=".as_slice()) {
        header.aspect *= atof(rest);
    } else if let Some(rest) = line.strip_prefix(b"PRIMARIES=".as_slice()) {
        let values = scan_floats::<8>(rest).ok_or_else(malformed)?;
        for (index, slot) in header.prims.iter_mut().flatten().enumerate() {
            *slot = values[index];
        }
    }
    Ok(())
}

/// `str2resolu` plus `scanlen`/`numscans` (`radiance.c:250-251`,
/// `radiance.c:276-305`): the axis written **second** is the scanline
/// length, and the `-`/`+` direction flags are parsed and then ignored:
/// libvips "will not rotate/flip as the FORMAT string asks"
/// (`radiance.c:70`).
fn parse_resolution(line: &[u8]) -> Result<(i64, i64), RadianceError> {
    let malformed = || RadianceError::BadResolution {
        line: String::from_utf8_lossy(&line[..line.len().min(80)]).into_owned(),
    };
    let last = |needle: u8| line.iter().rposition(|&b| b == needle);
    let (x_at, y_at) = match (last(b'X'), last(b'Y')) {
        (Some(x), Some(y)) => (x, y),
        _ => return Err(malformed()),
    };
    let x_extent = c_atoi(&line[x_at + 1..]);
    let y_extent = c_atoi(&line[y_at + 1..]);
    if x_extent <= 0 || y_extent <= 0 {
        return Err(malformed());
    }
    // `YMAJOR` is set when the X axis is written after the Y axis, and
    // `scanlen` then reads the X extent; otherwise the axes swap roles.
    if x_at > y_at {
        Ok((x_extent, y_extent))
    } else {
        Ok((y_extent, x_extent))
    }
}

/// C's `atoi`, saturating rather than overflowing.
fn c_atoi(text: &[u8]) -> i64 {
    let mut i = 0;
    while i < text.len() && text[i].is_ascii_whitespace() {
        i += 1;
    }
    let negative = match text.get(i) {
        Some(b'-') => {
            i += 1;
            true
        }
        Some(b'+') => {
            i += 1;
            false
        }
        _ => false,
    };
    let mut value: i64 = 0;
    while i < text.len() && text[i].is_ascii_digit() {
        value = value
            .saturating_mul(10)
            .saturating_add(i64::from(text[i] - b'0'));
        i += 1;
    }
    if negative { -value } else { value }
}

/// One scanline, run-length encoded or flat (`scanline_read`,
/// `radiance.c:404-484`).
fn read_scanline(
    cursor: &mut Cursor<'_>,
    scanline: &mut [[u8; 4]],
    row: u32,
) -> Result<(), RadianceError> {
    let width = scanline.len();
    let declared_width = width as u32;
    if !uses_run_length(declared_width) || cursor.peek() != Some(2) {
        return read_scanline_old(cursor, scanline, 0, row);
    }
    if cursor.remaining() < 4 {
        return Err(RadianceError::TruncatedScanline { row });
    }
    let head = cursor
        .quad()
        .ok_or(RadianceError::TruncatedScanline { row })?;
    scanline[0] = head;
    if head[1] != 2 || head[2] & 128 != 0 {
        return read_scanline_old(cursor, scanline, 1, row);
    }
    let declared = (u32::from(head[2]) << 8) | u32::from(head[3]);
    if declared != declared_width {
        return Err(RadianceError::ScanlineLengthMismatch {
            row,
            declared,
            width: declared_width,
        });
    }

    // Four separate component planes, red first, exponent last.
    for plane in 0..4 {
        let mut written = 0usize;
        while written < width {
            if cursor.remaining() < 2 {
                return Err(RadianceError::TruncatedScanline { row });
            }
            let code = cursor
                .byte()
                .ok_or(RadianceError::TruncatedScanline { row })?;
            let is_run = code > 128;
            let len = usize::from(if is_run { code & 127 } else { code });
            if written + len > width {
                return Err(RadianceError::ScanlineOverrun { row });
            }
            if is_run {
                let value = cursor
                    .byte()
                    .ok_or(RadianceError::TruncatedScanline { row })?;
                for slot in &mut scanline[written..written + len] {
                    slot[plane] = value;
                }
            } else {
                let bytes = cursor
                    .take(len)
                    .ok_or(RadianceError::TruncatedScanline { row })?;
                for (slot, value) in scanline[written..written + len].iter_mut().zip(bytes) {
                    slot[plane] = *value;
                }
            }
            written += len;
        }
    }
    Ok(())
}

/// The old-style scanline (`scanline_read_old`, `radiance.c:376-402`),
/// which is also what a sub-`MINELEN` or super-`MAXELEN` width uses.
///
/// `start` is where in `scanline` to begin, because the new-style reader
/// falls back to this after already consuming the first pixel.
fn read_scanline_old(
    cursor: &mut Cursor<'_>,
    scanline: &mut [[u8; 4]],
    start: usize,
    row: u32,
) -> Result<(), RadianceError> {
    let mut at = start;
    let mut shift = 0u32;
    while at < scanline.len() {
        let quad = cursor
            .quad()
            .ok_or(RadianceError::TruncatedScanline { row })?;
        if quad[0] == 1 && quad[1] == 1 && quad[2] == 1 {
            let Some(previous) = at.checked_sub(1).map(|i| scanline[i]) else {
                return Err(RadianceError::RepeatWithoutPixel { row });
            };
            let repeats = u32::from(quad[3]) << shift;
            for _ in 0..repeats {
                if at >= scanline.len() {
                    break;
                }
                scanline[at] = previous;
                at += 1;
            }
            // `radiance.c:392-397`: each chained marker shifts the count
            // eight bits further left, so four is the most vips accepts.
            // Without this the count grows without bound, which is the
            // `image` crate overflow in #539.
            shift += 8;
            if shift > 24 {
                return Err(RadianceError::RunawayRepeat { row });
            }
        } else {
            scanline[at] = quad;
            at += 1;
            shift = 0;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// One scanline (`scanline_write`, `radiance.c:955-978`).
///
/// `MINELEN` and `MAXELEN` choose an encoding here; they do not gate the
/// save. Measured on vips 8.18.4, a width-4 image writes a flat 4-bytes-
/// per-pixel payload and a width-40000 image does the same, both
/// successfully.
fn write_scanline(out: &mut Vec<u8>, scanline: &[[u8; 4]]) {
    let width = scanline.len();
    if !uses_run_length(width as u32) {
        for quad in scanline {
            out.extend_from_slice(quad);
        }
        return;
    }
    out.extend_from_slice(&[2, 2, (width >> 8) as u8, (width & 255) as u8]);
    for plane in 0..4 {
        let mut at = 0usize;
        while at < width {
            // Find the start and length of the next run worth coding
            // (`rle_scanline_write`, `radiance.c:508-524`).
            let mut run_at = at;
            let mut run_len = 1usize;
            while run_at < width {
                run_len = 1;
                while run_len < 127
                    && run_at + run_len < width
                    && scanline[run_at + run_len][plane] == scanline[run_at][plane]
                {
                    run_len += 1;
                }
                if run_len >= MINRUN {
                    break;
                }
                run_at += run_len;
            }
            // Everything before it goes out as literal blocks of up to 128.
            while at < run_at {
                let len = 128.min(run_at - at);
                out.push(len as u8);
                for quad in &scanline[at..at + len] {
                    out.push(quad[plane]);
                }
                at += len;
            }
            if run_len >= MINRUN {
                out.push(128 + run_len as u8);
                out.push(scanline[at][plane]);
                at += run_len;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroU16;

    /// The three-band float format every decode produces.
    fn float3() -> PixelFormat {
        PixelFormat::FloatF32(NonZeroU16::new(3).expect("3 is non-zero"))
    }

    /// Assemble a `.hdr` file from an explicit header, resolution line, and
    /// pixel payload, so a test can hand the decoder exactly the bytes it
    /// means to.
    fn hdr_file(header: &[&str], resolu: &str, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(b"#?RADIANCE\n");
        for line in header {
            out.extend_from_slice(line.as_bytes());
            out.push(b'\n');
        }
        out.push(b'\n');
        out.extend_from_slice(resolu.as_bytes());
        out.push(b'\n');
        out.extend_from_slice(payload);
        out
    }

    /// A standard `-Y h +X w` file whose payload is flat, unencoded RGBE.
    /// Only valid for `w < MINELEN`, which is what forces vips to write and
    /// read flat scanlines.
    fn flat_file(w: u32, h: u32, rgbe: &[[u8; 4]]) -> Vec<u8> {
        assert_eq!(rgbe.len(), (w * h) as usize, "pixel count must match wxh");
        let payload: Vec<u8> = rgbe.iter().flatten().copied().collect();
        hdr_file(
            &[&format!("FORMAT={FORMAT_RGBE}")],
            &format!("-Y {h} +X {w}"),
            &payload,
        )
    }

    /// The pixel payload of a `.hdr` file: everything after the blank line
    /// that ends the header and the resolution line that follows it.
    fn payload_of(file: &[u8]) -> &[u8] {
        let blank = file
            .windows(2)
            .position(|w| w == b"\n\n")
            .expect("header must end with a blank line");
        let after_header = &file[blank + 2..];
        let nl = after_header
            .iter()
            .position(|&b| b == b'\n')
            .expect("resolution line must end with a newline");
        &after_header[nl + 1..]
    }

    /// Build a `FloatF32(3)` raster from RGB triples, row-major.
    fn float_raster(w: u32, h: u32, px: &[[f32; 3]]) -> Raster {
        assert_eq!(px.len(), (w * h) as usize, "pixel count must match wxh");
        let mut data = Vec::with_capacity(px.len() * 12);
        for p in px {
            for c in p {
                data.extend_from_slice(&c.to_ne_bytes());
            }
        }
        Raster::new(w, h, float3(), data).expect("raster")
    }

    /// Pull the [`RadianceError`] out of a [`SourceError`], failing loudly
    /// on any other variant so a test cannot pass on the wrong error.
    fn radiance_error(e: SourceError) -> RadianceError {
        match e {
            SourceError::Radiance(r) => r,
            other => panic!("expected SourceError::Radiance, got {other:?}"),
        }
    }

    /**
     * Tests that the decoder uses libvips's half-bit-centred RGBE
     * constant `(mantissa + 0.5) * 2^(e - 136)` and its `exponent == 0`
     * hard-zero branch, rather than the plain `mantissa * 2^(e - 136)`
     * the `image` crate uses. Works by decoding a 6x1 flat-scanline file
     * (width below `MINELEN` 8, so the bytes on disk are the bytes the
     * decoder sees) whose pixels were captured from `vips getpoint` on
     * vips 8.18.4.
     * Input: RGBE `255,255,255,128` / `128,128,128,128` / `64,32,16,129` /
     * `255,0,0,140` / `0,0,0,0` / `1,2,3,0` -> Output: the exact values
     * vips printed, including `4088 8 8` where the plain form gives
     * `4080 0 0`, and `0 0 0` for both exponent-zero pixels.
     */
    #[test]
    fn decode_flat_scanline_matches_the_vips_half_bit_constant() {
        let px = [
            [255, 255, 255, 128],
            [128, 128, 128, 128],
            [64, 32, 16, 129],
            [255, 0, 0, 140],
            [0, 0, 0, 0],
            [1, 2, 3, 0],
        ];
        let raster = decode_radiance(&flat_file(6, 1, &px), DecodeLimits::default())
            .expect("the 6x1 oracle file decodes");

        assert_eq!(raster.width(), 6);
        assert_eq!(raster.height(), 1);
        assert_eq!(raster.format(), float3());
        assert_eq!(raster.interpretation(), Interpretation::ScRgb);

        let expected: [[f64; 3]; 6] = [
            [0.998046875, 0.998046875, 0.998046875],
            [0.501953125, 0.501953125, 0.501953125],
            [0.50390625, 0.25390625, 0.12890625],
            [4088.0, 8.0, 8.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        for (x, want) in expected.iter().enumerate() {
            let got = raster.getpoint(x as u32, 0);
            assert_eq!(got.len(), 3, "a decoded .hdr pixel has three bands");
            for (band, (g, w)) in got.iter().zip(want).enumerate() {
                let tol = w.abs().max(1.0) * 1e-6;
                assert!(
                    (g - w).abs() <= tol,
                    "pixel {x} band {band}: got {g}, vips 8.18.4 printed {w}"
                );
            }
        }
    }

    /**
     * Tests that the magic line is matched exactly, as
     * `vips__rad_israd` does, so the near-miss `#?RGBE` is rejected
     * rather than half-decoded. Works by mutating the first line of an
     * otherwise valid file three ways.
     * Input: `#?RGBE`, `#?RADIANCEX`, and an empty file -> Output:
     * `RadianceError::BadMagic` for the first two, and a bad-magic or
     * truncated-header rejection for the empty one; never a raster.
     */
    #[test]
    fn magic_line_must_be_exactly_radiance() {
        let good = flat_file(2, 1, &[[1, 2, 3, 128], [4, 5, 6, 128]]);

        let mut rgbe = good.clone();
        rgbe.splice(0..MAGIC.len(), b"#?RGBE\0\0\0\0".iter().copied());
        rgbe.splice(0..10, b"#?RGBE\n".iter().copied());
        assert!(
            matches!(
                radiance_error(decode_radiance(&rgbe, DecodeLimits::default()).unwrap_err()),
                RadianceError::BadMagic { .. }
            ),
            "#?RGBE is not a Radiance magic; vips routes such a file elsewhere"
        );

        let mut long = good.clone();
        long.splice(0..0, b"#?RADIANCEX\n".iter().copied());
        assert!(matches!(
            radiance_error(decode_radiance(&long, DecodeLimits::default()).unwrap_err()),
            RadianceError::BadMagic { .. }
        ));

        assert!(decode_radiance(&[], DecodeLimits::default()).is_err());
    }

    /**
     * Tests that the resolution line's axis order sets which extent is the
     * scanline length, and that the `-`/`+` direction flags are parsed and
     * then ignored, matching `scanlen`/`numscans` and libvips's own note
     * that it "will not rotate/flip as the FORMAT string asks". Works by
     * writing the same six pixels under three different resolution lines
     * and reading back the geometry.
     * Input: `-Y 1 +X 6`, `+X 6 +Y 1`, `-Y 1 -X 6` -> Output: 6x1, 1x6,
     * and 6x1, matching vips 8.18.4 measured on the same three files.
     */
    #[test]
    fn resolution_line_orientation_matches_vips() {
        // Nothing here may read as `1 1 1`, which is an old-style repeat
        // marker rather than a pixel.
        let payload: Vec<u8> = (0..6u8)
            .flat_map(|i| [i + 10, i + 20, i + 30, 128])
            .collect();
        for (resolu, want_w, want_h) in [
            ("-Y 1 +X 6", 6, 1),
            ("+X 6 +Y 1", 1, 6),
            ("-Y 1 -X 6", 6, 1),
        ] {
            let file = hdr_file(&[&format!("FORMAT={FORMAT_RGBE}")], resolu, &payload);
            let raster = decode_radiance(&file, DecodeLimits::default())
                .unwrap_or_else(|e| panic!("{resolu:?} should decode: {e}"));
            assert_eq!(
                (raster.width(), raster.height()),
                (want_w, want_h),
                "{resolu:?} geometry"
            );
        }
    }

    /**
     * Tests the new-style run-length decoder against a scanline captured
     * verbatim from `vips radsave`, so both the four separate component
     * planes and the `128 + count` run encoding are pinned. Works by
     * feeding vips's own 23-byte payload for a 16-pixel scanline back in
     * and checking every decoded RGBE quadruple.
     * Input: `2 2 0 16 | 144 127 | 134 63 10 7 15 ... 79 | 144 31 |
     * 144 129` -> Output: sixteen pixels whose red is 127, blue 31,
     * exponent 129, and whose green runs 63 six times then steps
     * 7, 15, 23, ... 79.
     */
    #[test]
    fn decode_rle_scanline_matches_the_vips_encoder_output() {
        let payload: [u8; 23] = [
            2, 2, 0, 16, 144, 127, 134, 63, 10, 7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 144, 31,
            144, 129,
        ];
        let file = hdr_file(&[&format!("FORMAT={FORMAT_RGBE}")], "-Y 1 +X 16", &payload);
        let raster = decode_radiance(&file, DecodeLimits::default()).expect("vips's own RLE bytes");
        assert_eq!((raster.width(), raster.height()), (16, 1));

        let greens = [
            63u8, 63, 63, 63, 63, 63, 7, 15, 23, 31, 39, 47, 55, 63, 71, 79,
        ];
        for (x, want_g) in greens.iter().enumerate() {
            let got = raster.getpoint(x as u32, 0);
            // Re-encode the decoded float back to its mantissa to compare
            // against the byte the RLE stream carried.
            let scale = f64::from(2.0f32).powi(129 - EXP_BIAS);
            let mantissa = |v: f64| (v / scale - 0.5).round() as u8;
            assert_eq!(mantissa(got[0]), 127, "pixel {x} red");
            assert_eq!(mantissa(got[1]), *want_g, "pixel {x} green");
            assert_eq!(mantissa(got[2]), 31, "pixel {x} blue");
        }
    }

    /**
     * Tests the scanline-length check libvips makes at `radiance.c:437-440`
     * and the `image` crate never makes: an RLE scanline's own two-byte
     * length marker has to agree with the width the header declared, or the
     * stream has desynchronised. Works by taking vips's own 16-pixel
     * scanline and declaring it 15 pixels wide in the marker.
     * Input: a `2 2 0 15` marker on a 16-wide image -> Output:
     * `RadianceError::ScanlineLengthMismatch { declared: 15, width: 16 }`.
     */
    #[test]
    fn rle_scanline_length_marker_must_match_the_declared_width() {
        let mut payload: Vec<u8> = vec![
            2, 2, 0, 15, 144, 127, 134, 63, 10, 7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 144, 31,
            144, 129,
        ];
        payload.push(0);
        let file = hdr_file(&[&format!("FORMAT={FORMAT_RGBE}")], "-Y 1 +X 16", &payload);
        let err = radiance_error(decode_radiance(&file, DecodeLimits::default()).unwrap_err());
        assert!(
            matches!(
                err,
                RadianceError::ScanlineLengthMismatch {
                    row: 0,
                    declared: 15,
                    width: 16
                }
            ),
            "got {err:?}"
        );
    }

    /**
     * Tests the overrun guard at `radiance.c:451-454`: a run or literal
     * block that would write past the end of its scanline is a malformed
     * file, not a buffer to grow. Works by replacing the red plane's
     * 16-pixel run with a 100-pixel one on a 16-wide image.
     * Input: a `128 + 100` run code in the red plane of a 16-wide scanline
     * -> Output: `RadianceError::ScanlineOverrun { row: 0 }`.
     */
    #[test]
    fn rle_run_may_not_overrun_the_scanline() {
        let payload: Vec<u8> = vec![2, 2, 0, 16, 228, 127, 144, 63, 144, 31, 144, 129];
        let file = hdr_file(&[&format!("FORMAT={FORMAT_RGBE}")], "-Y 1 +X 16", &payload);
        let err = radiance_error(decode_radiance(&file, DecodeLimits::default()).unwrap_err());
        assert!(
            matches!(err, RadianceError::ScanlineOverrun { row: 0 }),
            "got {err:?}"
        );
    }

    /**
     * Tests the `rshift > 24` bail-out at `radiance.c:392-397` on the
     * exact 81-byte input from #539, where the `image` crate's equivalent
     * loop multiplies without bound (panicking in debug, spinning in
     * release). Four chained old-style repeat markers are the most vips
     * accepts; the fifth marker never gets read because the fourth already
     * pushes the shift past 24. Works by decoding that file and asserting
     * the typed rejection.
     * Input: `-Y 1 +X 4`, one ordinary pixel, then eight consecutive
     * `01 01 01 00` markers -> Output: `RadianceError::RunawayRepeat`,
     * with no panic and no unbounded loop.
     */
    #[test]
    fn chained_old_style_repeat_markers_are_rejected() {
        let mut payload = vec![0u8, 0, 0, 0];
        for _ in 0..8 {
            payload.extend_from_slice(&[1, 1, 1, 0]);
        }
        let file = hdr_file(&[&format!("FORMAT={FORMAT_RGBE}")], "-Y 1 +X 4", &payload);
        assert_eq!(file.len(), 81, "the #539 reproducer is 81 bytes");

        let err = radiance_error(decode_radiance(&file, DecodeLimits::default()).unwrap_err());
        assert!(
            matches!(err, RadianceError::RunawayRepeat { row: 0 }),
            "got {err:?}"
        );

        // Three chained markers are still inside what vips accepts, so the
        // guard has to be a ceiling and not a blanket ban.
        let mut ok_payload = vec![9u8, 9, 9, 128];
        for _ in 0..3 {
            ok_payload.extend_from_slice(&[1, 1, 1, 0]);
        }
        ok_payload.extend_from_slice(&[7, 7, 7, 128]);
        ok_payload.extend_from_slice(&[8, 8, 8, 128]);
        ok_payload.extend_from_slice(&[6, 6, 6, 128]);
        ok_payload.extend_from_slice(&[5, 5, 5, 128]);
        let ok = hdr_file(
            &[&format!("FORMAT={FORMAT_RGBE}")],
            "-Y 1 +X 5",
            &ok_payload,
        );
        assert!(
            decode_radiance(&ok, DecodeLimits::default()).is_ok(),
            "three chained zero-count markers are accepted by vips"
        );
    }

    /**
     * Tests that a repeat marker at the very start of a scanline is
     * rejected rather than read out of bounds. libvips copies from
     * `scanline[-1]` here (`radiance.c:387`), one pixel before its own
     * buffer. Works by opening a flat scanline with the marker.
     * Input: `-Y 1 +X 4` whose first quadruple is `01 01 01 02` ->
     * Output: `RadianceError::RepeatWithoutPixel { row: 0 }`.
     */
    #[test]
    fn old_style_repeat_marker_needs_a_preceding_pixel() {
        let payload = vec![1u8, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let file = hdr_file(&[&format!("FORMAT={FORMAT_RGBE}")], "-Y 1 +X 4", &payload);
        let err = radiance_error(decode_radiance(&file, DecodeLimits::default()).unwrap_err());
        assert!(
            matches!(err, RadianceError::RepeatWithoutPixel { row: 0 }),
            "got {err:?}"
        );
    }

    /**
     * Tests that pixel data ending early is a typed error rather than a
     * short or zero-filled raster, on both the flat and the run-length
     * path. Works by truncating two otherwise valid files.
     * Input: a 4-wide flat file missing its last pixel, and a 16-wide RLE
     * file cut off mid-plane -> Output: `RadianceError::TruncatedScanline`
     * from both.
     */
    #[test]
    fn truncated_pixel_data_is_rejected() {
        let short_flat = hdr_file(
            &[&format!("FORMAT={FORMAT_RGBE}")],
            "-Y 1 +X 4",
            &[1, 2, 3, 128, 4, 5, 6, 128, 7, 8, 9, 128],
        );
        let err =
            radiance_error(decode_radiance(&short_flat, DecodeLimits::default()).unwrap_err());
        assert!(
            matches!(err, RadianceError::TruncatedScanline { row: 0 }),
            "got {err:?}"
        );

        let short_rle = hdr_file(
            &[&format!("FORMAT={FORMAT_RGBE}")],
            "-Y 1 +X 16",
            &[2, 2, 0, 16, 144, 127, 134],
        );
        let err = radiance_error(decode_radiance(&short_rle, DecodeLimits::default()).unwrap_err());
        assert!(
            matches!(err, RadianceError::TruncatedScanline { row: 0 }),
            "got {err:?}"
        );
    }

    /**
     * Tests that the header's scalars are parsed and attached under the
     * same field names libvips uses, so a `.hdr` that libviprs loads and
     * saves keeps them. Works by writing every recognised header line with
     * a distinctive value and reading the fields back.
     * Input: `EXPOSURE=`, `COLORCORR=`, `PIXASPECT=`, `PRIMARIES=` ->
     * Output: `rad-expos`, `rad-colcor-r/g/b`, `rad-aspect`, and the eight
     * `rad-prims-*` fields, all as doubles.
     */
    #[test]
    fn header_scalars_land_under_the_vips_field_names() {
        let file = hdr_file(
            &[
                &format!("FORMAT={FORMAT_RGBE}"),
                "EXPOSURE=2.5",
                "COLORCORR= 1.5 2.5 3.5",
                "PIXASPECT=1.25",
                "PRIMARIES= 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8",
                "VIEW= -vp 0 0 0",
            ],
            "-Y 1 +X 2",
            &[1, 2, 3, 128, 4, 5, 6, 128],
        );
        let raster = decode_radiance(&file, DecodeLimits::default()).expect("decodes");

        let double = |name: &str| {
            raster
                .get_field(name)
                .unwrap_or_else(|| panic!("{name} should be set"))
                .as_f64()
        };
        // libvips stores the colour-correction factors and the primaries in
        // `float` arrays (`COLOR` and `RGBPRIMS`, `radiance.c:184-187`) and
        // only widens them to double when it publishes the field, so the
        // value a caller reads back is the f32 rounding. libviprs matches
        // that rather than publishing a value vips cannot produce, which is
        // why 0.1 comes back as 0.10000000149011612 and the tolerance here
        // is single-precision.
        let cases: [(&str, f64); 13] = [
            ("rad-expos", 2.5),
            ("rad-colcor-r", 1.5),
            ("rad-colcor-g", 2.5),
            ("rad-colcor-b", 3.5),
            ("rad-aspect", 1.25),
            ("rad-prims-rx", 0.1),
            ("rad-prims-ry", 0.2),
            ("rad-prims-gx", 0.3),
            ("rad-prims-gy", 0.4),
            ("rad-prims-bx", 0.5),
            ("rad-prims-by", 0.6),
            ("rad-prims-wx", 0.7),
            ("rad-prims-wy", 0.8),
        ];
        for (name, want) in cases {
            let got = double(name);
            let tolerance = if name == "rad-expos" || name == "rad-aspect" {
                1e-12
            } else {
                f64::from(f32::EPSILON) * want.abs()
            };
            assert!(
                (got - want).abs() <= tolerance,
                "{name}: got {got}, expected {want} within {tolerance}"
            );
        }
        // Always the `COLRFMT` default, whatever the file declared: the
        // `FORMAT=` line is never parsed. See
        // `the_format_line_is_ignored_exactly_as_vips_ignores_it`.
        assert_eq!(
            raster
                .get_field("rad-format")
                .expect("rad-format")
                .as_str()
                .to_string(),
            FORMAT_RGBE
        );
    }

    /**
     * Tests that the `FORMAT=` line is read past and ignored, which is what
     * vips does and NOT what `radiance.c:693-698` reads like it does. The
     * tag arm there is unreachable: `radiance.c:636` calls
     * `formatval(line, read->format)` while `radiance.c:314` declares
     * `formatval(char fmt[MAXFMTLEN], const char *s)` with `fmt` as the
     * output buffer, so the arguments are swapped, the header line is never
     * parsed, and `read->format` keeps the `COLRFMT` default from
     * `radiance.c:610`. Measured on vips 8.18.4, a file declaring
     * `FORMAT=32-bit_rle_xyze` reports `rad-format: 32-bit_rle_rgbe` and
     * `interpretation: scrgb`; this pins libviprs to the same answer.
     *
     * Matching the reference here is deliberate rather than inherited,
     * because the interpretation tag reaches pixels: `colourspace()` reads
     * it, so tagging `Xyz` would make an XYZE file convert to sRGB
     * differently in libviprs than in vips. **If upstream fixes
     * `formatval`, libviprs should follow and start honouring `FORMAT=`**,
     * and this test is the thing that will fail when someone tries.
     *
     * Works by decoding the same pixels under three `FORMAT=` values, plus
     * a file with no `FORMAT=` line at all.
     * Input: rgbe, xyze, a nonsense format, and no format line -> Output:
     * `ScRgb` and `rad-format: 32-bit_rle_rgbe` from every one of them.
     */
    #[test]
    fn the_format_line_is_ignored_exactly_as_vips_ignores_it() {
        let mut headers: Vec<Vec<String>> = vec![
            vec![format!("FORMAT={FORMAT_RGBE}")],
            vec![format!("FORMAT={FORMAT_XYZE}")],
            vec!["FORMAT=something_else".to_string()],
            vec![],
        ];
        for header in &mut headers {
            let lines: Vec<&str> = header.iter().map(String::as_str).collect();
            let file = hdr_file(&lines, "-Y 1 +X 2", &[1, 2, 3, 128, 4, 5, 6, 128]);
            let raster = decode_radiance(&file, DecodeLimits::default()).expect("decodes");
            assert_eq!(
                raster.interpretation(),
                Interpretation::ScRgb,
                "vips tags every .hdr scRGB, whatever {lines:?} declares"
            );
            assert_eq!(
                raster.get_field("rad-format").expect("rad-format").as_str(),
                FORMAT_RGBE,
                "the COLRFMT default is what vips publishes, for {lines:?}"
            );
        }
    }

    /**
     * Tests that the save side still writes `FORMAT=32-bit_rle_xyze` for an
     * `Xyz` raster. Unlike the load side, `vips2rad_make_header`'s
     * interpretation override at `radiance.c:899-901` is live: it reads
     * `in->Type` rather than a parsed header line, so there is nothing
     * broken about it and nothing to reproduce. Works by tagging a raster
     * `Xyz` and reading the header line back off the encoded bytes.
     * Input: a 6x1 `FloatF32(3)` raster tagged `Xyz` -> Output: a file
     * whose `FORMAT=` line is `32-bit_rle_xyze`, which vips will then load
     * back as scRGB because of the load-side defect above.
     */
    #[test]
    fn save_writes_the_xyze_format_line_for_an_xyz_raster() {
        let px: Vec<[f32; 3]> = (0..6).map(|i| [1.0, i as f32 / 8.0, 0.25]).collect();
        let mut raster = float_raster(6, 1, &px);
        raster.meta.interpretation = Some(Interpretation::Xyz);
        let file = raster
            .encode_radiance(SaveOptions::default())
            .expect("encodes");
        let text = String::from_utf8_lossy(&file[..file.len().min(256)]).to_string();
        assert!(
            text.contains(&format!("FORMAT={FORMAT_XYZE}\n")),
            "an Xyz raster writes the CIE format line: {text:?}"
        );
        assert_eq!(
            decode_radiance(&file, DecodeLimits::default())
                .expect("decodes")
                .interpretation(),
            Interpretation::ScRgb,
            "and reading it back gives scRGB, because the load side cannot see it"
        );
    }

    /**
     * Tests that the decode budget is applied to the header's declared
     * geometry before anything is allocated, so a tiny run-length-encoded
     * file cannot declare a huge image and get it. Works by decoding one
     * small file under three tightened budgets.
     * Input: a 6x1 file under `max_coord = 4`, `max_pixels = 3`, and
     * `max_alloc_bytes = 8` -> Output: `CoordLimitExceeded`,
     * `DimensionLimitExceeded`, and `RadianceError::AllocLimitExceeded`.
     */
    #[test]
    fn decode_enforces_the_decode_budget() {
        let px = [[1u8, 2, 3, 128]; 6];
        let file = flat_file(6, 1, &px);

        let err = decode_radiance(&file, DecodeLimits::default().with_max_coord(4)).unwrap_err();
        assert!(
            matches!(err, SourceError::CoordLimitExceeded { width: 6, .. }),
            "got {err:?}"
        );

        let err = decode_radiance(&file, DecodeLimits::default().with_max_pixels(3)).unwrap_err();
        assert!(
            matches!(err, SourceError::DimensionLimitExceeded { width: 6, .. }),
            "got {err:?}"
        );

        let err = radiance_error(
            decode_radiance(&file, DecodeLimits::default().with_max_alloc_bytes(8)).unwrap_err(),
        );
        assert!(
            matches!(
                err,
                RadianceError::AllocLimitExceeded {
                    needed: 72,
                    max_alloc_bytes: 8,
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    /**
     * Tests that the budget bites at exactly the byte the declared
     * geometry costs, and not one byte either side. The case above refuses
     * at a budget of eight against a price of seventy-two, which a price
     * wrong by a factor would also refuse; only the exact pair below pins
     * the arithmetic and the `>` in the comparison.
     * Input: the same 6x1 file at `max_alloc_bytes` 72 then 71 -> Output: a
     * clean 6x1 decode, then `AllocLimitExceeded { needed: 72 }`.
     */
    #[test]
    fn the_decode_budget_bites_at_exactly_the_declared_price() {
        let px = [[1u8, 2, 3, 128]; 6];
        let file = flat_file(6, 1, &px);

        let exact = DecodeLimits::default().with_max_alloc_bytes(72);
        let raster = decode_radiance(&file, exact).expect("72 bytes is exactly a 6x1 RGB float");
        assert_eq!((raster.width(), raster.height()), (6, 1));

        let short = DecodeLimits::default().with_max_alloc_bytes(71);
        let err = radiance_error(decode_radiance(&file, short).unwrap_err());
        assert!(
            matches!(
                err,
                RadianceError::AllocLimitExceeded {
                    width: 6,
                    height: 1,
                    needed: 72,
                    max_alloc_bytes: 71,
                }
            ),
            "got {err:?}"
        );
    }

    /**
     * Tests the encoder against `vips float2rad`'s `setcolr` byte for
     * byte, including the three constants that are easy to get subtly
     * wrong: the `1e-32` floor, `frexp` rather than `log2().floor() + 1`,
     * and the `255.9999` scale with a truncating rather than rounding
     * conversion. Works by encoding twelve float triples whose RGBE
     * quadruples were captured from vips 8.18.4 and comparing the flat
     * payload (width 6 is below `MINELEN`, so no run-length encoding
     * intervenes).
     * Input: twelve triples covering unity, zero, either side of the
     * `1e-32` floor, negatives, an exact power of two, and infinity ->
     * Output: the 48 payload bytes vips wrote.
     */
    // The oracle triples below are spelled out to their full exact binary
    // value rather than to the shortest form that round-trips as an f32,
    // because the point of several of them is their exact relationship to
    // `(mantissa + 0.5) / 256`; shortening them would hide it.
    #[allow(clippy::excessive_precision)]
    #[test]
    fn encode_matches_vips_float2rad_setcolr() {
        let px: [[f32; 3]; 12] = [
            [1.0, 0.5, 0.25],
            [0.0, 0.0, 0.0],
            [1e-33, 1e-33, 1e-33],
            [1e-31, 0.0, 0.0],
            [4088.0, 8.0, 8.0],
            [-1.0, 2.0, -3.0],
            [65504.0, 1.0, 0.001],
            [0.998046875, 0.501953125, 0.12890625],
            [f32::INFINITY, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [0.998046875, 0.998046875, 0.998046875],
            [3.0517578125e-05, 1.0, 1.0],
        ];
        let want: [u8; 48] = [
            127, 63, 31, 129, 0, 0, 0, 0, 0, 0, 0, 0, 129, 0, 0, 26, 255, 0, 0, 140, 0, 127, 0,
            130, 255, 0, 0, 144, 255, 128, 32, 128, 0, 0, 0, 128, 127, 127, 127, 130, 255, 255,
            255, 128, 0, 127, 127, 129,
        ];

        let raster = float_raster(6, 2, &px);
        let file = raster
            .encode_radiance(SaveOptions::default())
            .expect("a FloatF32(3) raster encodes");
        assert_eq!(
            payload_of(&file),
            &want[..],
            "the RGBE payload must match vips float2rad byte for byte"
        );
    }

    /**
     * Tests that the encoder run-length encodes inside libvips's
     * `MINELEN..=MAXELEN` window, using its four separate component planes
     * and its `128 + count` run code. Works by encoding the 16-pixel
     * scanline whose 23-byte payload was captured from `vips radsave`.
     * Input: six identical pixels then a green ramp, at width 16 ->
     * Output: `2 2 0 16 | 144 127 | 134 63 10 7 ... 79 | 144 31 |
     * 144 129`, exactly what vips wrote.
     */
    #[test]
    fn encode_run_length_encodes_inside_the_vips_size_range() {
        let mut px = Vec::new();
        for i in 0..16u32 {
            let g = if i < 6 { 0.5 } else { (i - 5) as f32 / 16.0 };
            px.push([1.0f32, g, 0.25]);
        }
        let want: [u8; 23] = [
            2, 2, 0, 16, 144, 127, 134, 63, 10, 7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 144, 31,
            144, 129,
        ];
        let file = float_raster(16, 1, &px)
            .encode_radiance(SaveOptions::default())
            .expect("encodes");
        assert_eq!(payload_of(&file), &want[..]);
    }

    /**
     * Tests that `MINELEN` and `MAXELEN` select an encoding rather than
     * gating the save, correcting a widespread misreading of
     * `radiance.c:955-978`. Measured on vips 8.18.4, a width-4 image saves
     * with a 32-byte flat payload for two rows and a width-40000 image
     * saves with a 320000-byte flat payload, both exiting 0. Works by
     * encoding at width 4 and checking the payload is exactly four bytes
     * per pixel with no run markers.
     * Input: a 4x2 float raster -> Output: 32 payload bytes, and the same
     * raster back after a decode.
     */
    #[test]
    fn encode_writes_flat_scanlines_below_the_size_range() {
        let px: Vec<[f32; 3]> = (0..8).map(|i| [1.0, i as f32 / 8.0, 0.25]).collect();
        let raster = float_raster(4, 2, &px);
        let file = raster
            .encode_radiance(SaveOptions::default())
            .expect("encodes");
        assert_eq!(
            payload_of(&file).len(),
            4 * 4 * 2,
            "below MINELEN vips writes flat scanlines, four bytes per pixel"
        );
        let back = decode_radiance(&file, DecodeLimits::default()).expect("round trip");
        assert_eq!((back.width(), back.height()), (4, 2));
    }

    /**
     * Tests the exact fixed point of the encode/decode pair: re-encoding a
     * decoded RGBE quadruple reproduces it byte for byte when its largest
     * mantissa is at least 128 and its exponent byte is in `23..=255`,
     * which is precisely the normalised form any encoder emits. Works by
     * building one flat file holding a wide sample of such quadruples,
     * decoding it, re-encoding it, and comparing payloads.
     * Input: a 6-wide flat file of normalised quadruples across the
     * exponent range -> Output: an identical payload, so `float2rad` after
     * `rad2float` is the identity there.
     */
    #[test]
    fn the_encode_decode_pair_is_the_identity_on_normalised_quadruples() {
        let mut quads: Vec<[u8; 4]> = Vec::new();
        for e in [23u8, 24, 40, 64, 100, 128, 136, 180, 200, 254, 255] {
            for mx in [128u8, 129, 170, 200, 254, 255] {
                for other in [0u8, 1, mx / 3, mx / 2, mx - 1, mx] {
                    quads.push([mx, other, (other / 2).min(mx), e]);
                    quads.push([other, mx, (other / 3).min(mx), e]);
                }
            }
        }
        while !quads.len().is_multiple_of(6) {
            quads.push([255, 128, 128, 128]);
        }
        let h = (quads.len() / 6) as u32;
        let file = flat_file(6, h, &quads);
        let raster = decode_radiance(&file, DecodeLimits::default()).expect("decodes");
        let again = raster
            .encode_radiance(SaveOptions::default())
            .expect("re-encodes");
        assert_eq!(
            payload_of(&again),
            payload_of(&file),
            "float2rad after rad2float must be the identity on normalised RGBE"
        );
    }

    /**
     * Tests that the encoder rejects everything except `FloatF32(3)`
     * rather than silently inventing a colourspace policy for it. vips
     * advertises `radsave` as accepting mono, but measured on 8.18.4 a
     * one-band image fails with "float2rad: image must have at least 3
     * bands", so there is no mono behaviour to match.
     * Input: `Rgb8`, `FloatF32(1)`, and `RgbaF32` rasters -> Output:
     * `EncodeError::Encode` from each, with no bytes produced.
     */
    #[test]
    fn encode_rejects_anything_but_three_band_float() {
        let rgb8 = Raster::new(4, 1, PixelFormat::Rgb8, vec![9u8; 12]).unwrap();
        let mono = Raster::new(
            4,
            1,
            PixelFormat::FloatF32(NonZeroU16::new(1).unwrap()),
            vec![0u8; 16],
        )
        .unwrap();
        let rgba = Raster::new(4, 1, PixelFormat::RgbaF32, vec![0u8; 64]).unwrap();
        for raster in [rgb8, mono, rgba] {
            let fmt = raster.format();
            match raster.encode_radiance(SaveOptions::default()) {
                Err(EncodeError::Encode(_)) => {}
                other => panic!("{fmt:?} should be rejected, got {other:?}"),
            }
        }
    }

    /**
     * Tests that a saved file carries the header vips writes and that the
     * save/load pair round-trips through the filesystem, including the
     * header scalars the options and the raster's fields supply. Works by
     * saving a small raster to a temporary file and decoding it back.
     * Input: a 6x1 `FloatF32(3)` raster and `SaveOptions` overriding the
     * exposure -> Output: a file opening `#?RADIANCE` with
     * `FORMAT=32-bit_rle_rgbe`, a `-Y 1 +X 6` resolution line, and a
     * decode reporting `rad-expos` 4.
     */
    #[test]
    fn save_radiance_writes_a_file_decode_radiance_reads_back() {
        let px: Vec<[f32; 3]> = (0..6).map(|i| [1.0, i as f32 / 8.0, 0.25]).collect();
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("out.hdr");
        float_raster(6, 1, &px)
            .save_radiance(
                &path,
                SaveOptions {
                    exposure: Some(4.0),
                    ..Default::default()
                },
            )
            .expect("saves");

        let bytes = std::fs::read(&path).expect("reads back");
        let text = String::from_utf8_lossy(&bytes[..bytes.len().min(256)]).to_string();
        assert!(text.starts_with("#?RADIANCE\n"), "header: {text:?}");
        assert!(
            text.contains(&format!("FORMAT={FORMAT_RGBE}\n")),
            "{text:?}"
        );
        assert!(text.contains("-Y        1 +X        6\n"), "{text:?}");

        let back = decode_radiance(&bytes, DecodeLimits::default()).expect("decodes");
        assert_eq!((back.width(), back.height()), (6, 1));
        assert_eq!(back.interpretation(), Interpretation::ScRgb);
        let expos = back.get_field("rad-expos").expect("rad-expos").as_f64();
        assert!((expos - 4.0).abs() <= 1e-9, "got {expos}");
    }

    /**
     * Tests that the header scalars a save reads back off the raster
     * cannot break the file, whatever a caller wrote into them. Every one
     * of these fields is public through `Raster::set_field`, the typed
     * accessors on `MetadataValue` panic on a mismatched variant, and a
     * newline inside `rad-format` would inject header lines into the
     * output. Works by setting a field of the wrong shape and a field
     * carrying a newline, then saving and reading the result back.
     * Input: `rad-expos` as a string and `rad-format` carrying
     * `"x\n\n-Y 9 +X 9"` -> Output: a save that succeeds, a header with
     * one blank line in it, and a decode that reports the original
     * geometry.
     */
    #[test]
    fn hostile_header_fields_cannot_corrupt_a_save() {
        let px: Vec<[f32; 3]> = (0..6).map(|i| [1.0, i as f32 / 8.0, 0.25]).collect();
        let mut raster = float_raster(6, 1, &px);
        raster.set_field("rad-expos", "not a number".into());
        raster.set_field("rad-aspect", MetadataValue::Blob(vec![1, 2, 3]));
        raster.meta.interpretation = Some(Interpretation::Multiband);
        raster.set_field("rad-format", "x\n\n-Y 9 +X 9".into());

        let file = raster
            .encode_radiance(SaveOptions::default())
            .expect("a hostile field is a fallback, not a failure");
        assert_eq!(
            file.windows(2).filter(|w| *w == b"\n\n").count(),
            1,
            "exactly one blank line, the one that ends the header"
        );
        let back = decode_radiance(&file, DecodeLimits::default()).expect("decodes");
        assert_eq!(
            (back.width(), back.height()),
            (6, 1),
            "the injected `-Y 9 +X 9` must not have become the resolution line"
        );
        // Not a byte comparison: an arbitrary float is not an RGBE fixed
        // point, so one pass through the codec quantises it. 1.0 encodes to
        // mantissa 127 and decodes to 127.5/128.
        let first = back.getpoint(0, 0);
        assert!(
            (first[0] - 0.99609375).abs() <= 1e-9,
            "red should quantise to 127.5/128, got {}",
            first[0]
        );
    }

    /**
     * Sweeps the seeded fuzz corpus through the decoder, so every
     * malformation it holds is a `cargo test` regression rather than
     * something only a fuzz run would notice. The 81-byte #539 reproducer
     * is the headline entry: the `image` crate panics on it in debug and
     * spins on it in release. Works by decoding every file under
     * `fuzz/corpus/fuzz_radiance/` and checking each against the outcome
     * its name promises.
     * Input: sixteen corpus files -> Output: the named typed error from
     * each malformed one, a raster from each valid one, and no panic from
     * any of them.
     */
    #[test]
    fn the_fuzz_corpus_decodes_or_fails_exactly_as_named() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fuzz")
            .join("corpus")
            .join("fuzz_radiance");
        let limits = DecodeLimits::default().with_max_alloc_bytes(4 * 1024 * 1024);

        let mut seen = 0usize;
        for entry in std::fs::read_dir(&dir).expect("the seeded corpus is in the tree") {
            let path = entry.expect("corpus entry").path();
            let name = path
                .file_name()
                .expect("corpus entries are files")
                .to_string_lossy()
                .into_owned();
            let bytes = std::fs::read(&path).expect("corpus file");
            let result = decode_radiance(&bytes, limits);
            seen += 1;

            let ok = match name.as_str() {
                "issue-539-chained-old-rle-markers" => {
                    assert_eq!(bytes.len(), 81, "the #539 reproducer is 81 bytes");
                    matches!(
                        result,
                        Err(SourceError::Radiance(RadianceError::RunawayRepeat { .. }))
                    )
                }
                "rle-length-marker-mismatch" => matches!(
                    result,
                    Err(SourceError::Radiance(
                        RadianceError::ScanlineLengthMismatch { .. }
                    ))
                ),
                "rle-run-overruns-scanline" => matches!(
                    result,
                    Err(SourceError::Radiance(RadianceError::ScanlineOverrun { .. }))
                ),
                "old-rle-marker-without-pixel" => matches!(
                    result,
                    Err(SourceError::Radiance(
                        RadianceError::RepeatWithoutPixel { .. }
                    ))
                ),
                "truncated-pixel-data" => matches!(
                    result,
                    Err(SourceError::Radiance(
                        RadianceError::TruncatedScanline { .. }
                    ))
                ),
                "header-without-blank-line" | "magic-only" => matches!(
                    result,
                    Err(SourceError::Radiance(RadianceError::TruncatedHeader { .. }))
                ),
                "unparseable-resolution" => matches!(
                    result,
                    Err(SourceError::Radiance(RadianceError::BadResolution { .. }))
                ),
                "resolution-out-of-bounds" => matches!(
                    result,
                    Err(SourceError::Radiance(
                        RadianceError::DimensionOutOfBounds { .. }
                    ))
                ),
                "rle-bomb" => matches!(
                    result,
                    Err(SourceError::Radiance(
                        RadianceError::AllocLimitExceeded { .. }
                    ))
                ),
                "empty" => matches!(
                    result,
                    Err(SourceError::Radiance(RadianceError::BadMagic { .. }))
                ),
                _ => {
                    assert!(
                        name.starts_with("valid-")
                            || name == "dos-line-endings"
                            || name == "x-major-resolution",
                        "unclassified corpus entry {name}"
                    );
                    result.is_ok()
                }
            };
            assert!(ok, "corpus entry {name}: got {result:?}");
        }
        assert_eq!(seen, 16, "the corpus should still hold sixteen seeds");
    }

    /**
     * Pins the shape of the options struct: both knobs default to "take it
     * from the raster, else the Radiance default", and the struct is open
     * enough to build with `..Default::default()` from outside its own
     * module. Works by comparing `SaveOptions::default()` against an
     * explicit literal and a functional-update literal.
     * Input: none -> Output: all three spellings compare equal, with both
     * fields `None`.
     */
    #[test]
    fn save_options_default_to_the_rasters_own_header_scalars() {
        let explicit = SaveOptions {
            exposure: None,
            aspect: None,
        };
        let updated = SaveOptions {
            ..Default::default()
        };
        assert_eq!(SaveOptions::default(), explicit);
        assert_eq!(updated, explicit);
    }
}
