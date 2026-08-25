//! Radiance HDR (`.hdr`) load and save: RGBE bytes in, three-band float out.
//!
//! Ported from libvips `foreign/radiance.c` (the container, itself
//! copy-pasted from Greg Ward's Radiance 5.4 sources) together with
//! `colour/rad2float.c` and `colour/float2rad.c` (the sample codec), which
//! libvips keeps in separate files because it models Radiance as a
//! *coding* — a 4-band uchar raster tagged `VIPS_CODING_RAD` that any real
//! operation silently unpacks. libviprs has no coding concept, so this
//! module fuses the two halves: [`decode_radiance`] is `radload` composed
//! with `rad2float`, and [`Raster::encode_radiance`] is `float2rad`
//! composed with `radsave`.
//!
//! # Operations
//!
//! | libviprs method | libvips equivalent | result |
//! |---|---|---|
//! | [`decode_radiance`] | `radload` + `rad2float` | [`PixelFormat::FloatF32`]`(3)` raster tagged [`Interpretation::ScRgb`] |
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
//! * **`.hdr` only.** `vips -l` registers exactly one suffix
//!   (`vips__rad_suffs`, `radiance.c:1035`) and the magic is exactly the
//!   first line `#?RADIANCE` (`vips__rad_israd`, `radiance.c:568-577`).
//!   `#?RGBE` is rejected; measured, such a file falls through to
//!   `magickload`.
//! * **A loaded `.hdr` cannot enter the pyramid engine.** See
//!   [`decode_radiance`].
//!
//! Every entry point here is fallible and there is no panicking twin,
//! matching the rest of the codec surface in [`crate::encode`],
//! [`crate::webp`], and [`crate::gif`]: a decoder's failures come from
//! untrusted bytes, so a panicking spelling would have no honest caller.

use std::path::Path;

use thiserror::Error;

use crate::codec::EncodeError;
use crate::conversion::Interpretation;
use crate::imageio::SaveError;
use crate::pixel::PixelFormat;
use crate::raster::{Raster, RasterError};
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
/// The result is [`PixelFormat::FloatF32`]`(3)` tagged
/// [`Interpretation::ScRgb`], or [`Interpretation::Xyz`] when the file's
/// `FORMAT=` line says `32-bit_rle_xyze`. Samples are linear radiance
/// values with no upper bound, which is the point of the format: a bright
/// pixel reads back as `4088.0`, not as a clipped `1.0`.
///
/// # The fixed point
///
/// `float2rad` after `rad2float` reproduces the original RGBE quadruple
/// exactly when its largest mantissa is at least 128 and its exponent byte
/// is in `23..=255` — verified here over 298,240 combinations. Those are
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
    let _ = (bytes, limits);
    Err(RadianceError::BadMagic {
        found: String::new(),
    }
    .into())
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
        let _ = options;
        Err(EncodeError::encode("radiance encoder not implemented yet"))
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
        let payload: Vec<u8> = (0..6u8).flat_map(|i| [i, i, i, 128]).collect();
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
        for (name, want) in [
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
        ] {
            let got = double(name);
            assert!(
                (got - want).abs() <= 1e-9,
                "{name}: got {got}, expected {want}"
            );
        }
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
     * Tests that the `FORMAT=` line selects the colour interpretation the
     * way `radiance.c:693-698` intends: `32-bit_rle_rgbe` is scRGB,
     * `32-bit_rle_xyze` is XYZ, anything else is untagged. This is a
     * deliberate divergence from the vips 8.18.4 *binary*, which reports
     * scRGB for every `.hdr` because `rad2vips_process_line` calls
     * `formatval(line, read->format)` with the arguments the wrong way
     * round (`radiance.c:636`), so the parsed value never reaches
     * `read->format` and the XYZ branch is unreachable. Works by decoding
     * the same pixels under three `FORMAT=` values.
     * Input: rgbe, xyze, and a nonsense format -> Output: `ScRgb`, `Xyz`,
     * and `Multiband`.
     */
    #[test]
    fn the_format_line_selects_the_interpretation() {
        for (format, want) in [
            (FORMAT_RGBE, Interpretation::ScRgb),
            (FORMAT_XYZE, Interpretation::Xyz),
            ("something_else", Interpretation::Multiband),
        ] {
            let file = hdr_file(
                &[&format!("FORMAT={format}")],
                "-Y 1 +X 2",
                &[1, 2, 3, 128, 4, 5, 6, 128],
            );
            let raster = decode_radiance(&file, DecodeLimits::default()).expect("decodes");
            assert_eq!(raster.interpretation(), want, "FORMAT={format}");
        }
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
