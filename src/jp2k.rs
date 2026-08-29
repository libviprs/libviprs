//! JPEG 2000 (`.jp2`, `.j2k`) load and save: every codestream this build can
//! read in, a JP2 container out.
//!
//! Ported from libvips `foreign/jp2kload.c` and `foreign/jp2ksave.c`. libvips
//! wraps OpenJPEG and so reaches the whole format in both directions; libviprs
//! reaches it through two pure-Rust crates that split the job the way
//! [`crate::jxl`]'s two do, and on the same line. `hayro-jpeg2000` decodes,
//! because it is `#![forbid(unsafe_code)]` at the settings this build uses and
//! the decoder is the half that eats attacker-controlled bytes.
//! `openjpeg2-pure-rs` encodes, because it is a translation of the reference
//! C and the encoder only ever sees a [`Raster`] this crate already owns.
//!
//! Both sit behind the non-default **`jp2k`** feature, the way `resvg` sits
//! behind `svg`, but for a different reason and at a very different price:
//! **+2 lock entries, and both of them are these two crates**, because neither
//! has a dependency of its own. `svg` costs +29 and `jxl` costs +21. What is
//! left is the compile, 9.7k lines of decoder and 36.9k of encoder, which
//! nobody who does not read or write JPEG 2000 should pay for. Without the
//! feature every entry point below still exists, still compiles and keeps its
//! signature; each returns a typed refusal naming the feature, so a caller
//! compiles against either build. [`Jp2kError`] and the
//! [`SourceError::Jp2k`] variant carrying it are declared in both builds for
//! the same reason, and [`Jp2kError::FeatureNotEnabled`] is the arm the
//! feature-off build takes.
//!
//! # Operations
//!
//! | libviprs method | libvips equivalent | result |
//! |---|---|---|
//! | [`decode_jp2k`] | `jp2kload` / `jp2kload_buffer` (default `page = 0`) | 8-bit or 16-bit raster at the image origin's offset, plus `icc-profile-data` / `bits-per-sample` / `jp2k-resolutions` / `tile-width` / `tile-height` |
//! | [`Raster::encode_jp2k`] | `jp2ksave_buffer` | `.jp2` bytes, always in a JP2 container |
//! | [`Raster::save_jp2k`] | `jp2ksave` | `.jp2` file |
//!
//! # Semantics
//!
//! Everything below was measured against `/opt/homebrew/bin/vips` 8.18.6,
//! linking libopenjp2 2.5.4, over the 27 fixtures in
//! `oracle-captures/foreign-jp2k/`. The numbers are in
//! `oracle-captures/foreign-jp2k/oracle.json`.
//!
//! * **Both carriers are read and the reversible path is exact.** Of the 22
//!   fixtures the decoder accepts, **fourteen are byte-identical** to what
//!   `vips rawsave` wrote: every reversible 5/3 file, at every component
//!   precision from 2 to 16 bits, greyscale, RGB, RGBA and CMYK, tiled and
//!   untiled, subsampled and multi-resolution. The reversible wavelet is
//!   integer-specified, so that is a parity port rather than an approximation
//!   and its pins carry no tolerance at all. Of the other eight, four are the
//!   irreversible fixtures below, three are refused on carrier grounds (one
//!   signed component and two 31-bit ones), and one is `origin57.j2k`, whose
//!   geometry diverges and is issue #766.
//! * **The irreversible path needs a tolerance, and it is 4.** The 9/7
//!   wavelet is float-specified, so `hayro-jpeg2000` and OpenJPEG are entitled
//!   to disagree in the last place. Measured, they disagree by at most **4
//!   counts** on the four lossy fixtures: `rgb_lossy_q48` at Q 48 is the worst
//!   at 4, `chroma_sub_on` at Q 90 reaches 3, `chroma_sub_off` reaches 2 and
//!   `chroma_tiny_sub_on` reaches 1. The pins carry exactly that and no more,
//!   per fixture rather than as a shared slack number. Rounding rather than
//!   truncating the reconstructed sample is what buys the difference between a
//!   maximum of 4 and a maximum of 3 becoming 4 in more places: measured, it
//!   is never worse and is better on two of the four.
//! * **Bit depth is left-justified, exactly as `jp2kload` does it.** A
//!   precision-`N` component is shifted left by `element_bits - N`
//!   (`vips_foreign_load_jp2k_ljust`), so a 12-bit sample of 4095 comes back
//!   as 65520 rather than 4095 or 65535, and the true depth survives only in
//!   `bits-per-sample`. Depths 8 and 16 shift by zero, which is exactly why an
//!   8-bit-only test cannot catch a port that forgets the shift; the depth
//!   sweep here runs 2, 4, 8, 10, 12, 14 and 16 and every one of them is
//!   pinned to the vips answer.
//! * **A signed component is refused rather than silently offset.** libvips
//!   has `char` and `short` band formats and hands a signed component to one
//!   of them; [`PixelFormat`] has no signed carrier at all, and
//!   `hayro-jpeg2000` does not report signedness, returning every component
//!   DC-level-shifted into the unsigned range. So a signed file decoded
//!   naively comes back offset by exactly half the range: measured on
//!   `depth12s.j2k`, libviprs would say 0 where vips says -32768. Rather than
//!   ship that, [`decode_jp2k`] reads the sign bit out of the codestream's own
//!   SIZ marker and refuses with [`Jp2kError::SignedComponent`].
//! * **More than 16 bits of precision is refused too, for two reasons at
//!   once.** [`PixelFormat`] has no 32-bit integer carrier, and
//!   `hayro-jpeg2000` hands samples back as `f32`, whose 24-bit mantissa
//!   cannot hold a 31-bit sample: measured on `int31.jp2`, three distinct
//!   input values all come back as the same float. vips's own answer there is
//!   not a round trip either (`jp2ksave` writes 31 bits for a 4-byte format
//!   and `jp2kload` doubles them coming back), so there is nothing to be
//!   faithful to. [`Jp2kError::PrecisionNotSupported`] names the ceiling.
//! * **A bare codestream with subsampled chroma gets the inverse YCC, and it
//!   is OpenJPEG's arithmetic and not a textbook matrix.** `jp2kload` treats
//!   an unspecified colour space with three components and subsampling on
//!   components 1 and 2 as YCC and runs `sycc_to_rgb`
//!   (`vips_foreign_load_jp2k_get_ycc`); `hayro-jpeg2000` hands the raw
//!   planes back instead, so `sub420.j2k` would decode to a completely
//!   different picture (measured: 240 counts out on the first pixel). This
//!   module runs the transform itself, with OpenJPEG's exact coefficients and
//!   its truncating `(int)` casts, which is what makes the fixture land on
//!   vips's pixels rather than one off them: a rounding implementation gets
//!   `[243, 98, 0]` where the file says `[242, 98, 0]`.
//! * **Save always writes a JP2 container.** `jp2ksave` registers `.j2k`,
//!   `.jp2`, `.jpt`, `.j2c` and `.jpc` and writes the same RFC 3745 container
//!   for all five, byte for byte, because it hard-codes `OPJ_CODEC_JP2`.
//!   [`Raster::encode_jp2k`] does the same and takes no carrier argument, so
//!   a port cannot pick one from an extension. The loader is the opposite and
//!   sniffs, so a bare codestream still reads.
//! * **The resolution count is `jp2k-resolutions`, not `n-pages`.**
//!   `vipsheader` calls it `n-pages` and vips's `page` argument selects a
//!   resolution level rather than a frame, so `[page=1]` is the same picture
//!   at half size. This crate reserves `n-pages` for counts a zero-based
//!   `page` argument can select (issue #635), and [`decode_jp2k`] has no such
//!   argument yet, so the count travels under its own key the way an OpenEXR
//!   part count travels as `exr-parts`. Reading a level other than 0 waits on
//!   the page model in #564.
//! * **Save reproduces vips's resolution count exactly.** `jp2ksave` sets
//!   `numresolution` to `max(1, floor(log2(min(width, height))) - 5)`.
//!   Measured over ten sizes, `floor` is right and `ceil` is wrong at four of
//!   them (65, 100, 129 and 1000), so the formula is pinned from the binary
//!   rather than read off the source.
//! * **Metadata goes one way only, and that is vips's shape.** `jp2ksave`
//!   inherits `--profile` and `--keep` from `VipsForeignSave` and implements
//!   neither: there is no ICC, EXIF, XMP or IPTC code in `jp2ksave.c` at all,
//!   and saving with a profile produces a byte-identical file. So
//!   [`SaveOptions`] has no `keep` field and
//!   [`Raster::save`](crate::Raster::save) and
//!   [`Raster::save_stripped`](crate::Raster::save_stripped) would write
//!   identical bytes. `jp2kload` does read an ICC profile, out of a `METH=2`
//!   `colr` box, and copies the payload verbatim without validating it;
//!   [`decode_jp2k`] walks the boxes for that itself, because
//!   `hayro-jpeg2000` drops a profile it cannot parse and the oracle fixture
//!   `icc_colr.jp2` carries a deliberately invalid 24-byte one.
//! * **Lossy is a compression ratio, not a `Q`.** `jp2ksave --Q` sets
//!   OpenJPEG's `cp_fixed_quality` with a per-layer `tcp_distoratio`, which is
//!   a distortion ratio in decibels. `openjpeg2-pure-rs` exposes `rates`,
//!   which is `cp_disto_alloc` with a per-layer compression ratio, and its
//!   internals are `pub(crate)` so nothing else is reachable. Those are
//!   different numbers, so [`Compression::Lossy`] carries a `ratio` and there
//!   is no `Q` field for this module to accept and reinterpret. That is the
//!   same answer [`crate::jxl`] gave to `jxlsave`'s `distance`, and for the
//!   same reason: an argument the encoder reads as something else is worse
//!   than no argument at all.
//! * **Float is refused, matching vips.** `vips jp2ksave` on a `float` or
//!   `double` image fails with `not an integer format`; measured, and
//!   [`Raster::encode_jp2k`] refuses the float carriers rather than casting.
//! * **A `colr` box is what makes a colour space specified.** The YCC
//!   condition above is "bare codestream *and* subsampled", not "subsampled",
//!   because a JP2 always carries a `colr` box and so is never unspecified by
//!   construction. Measured on the same subsampled codestream wrapped three
//!   ways: `EnumCS 16` gives `[128, 16, 240]` and no transform, `EnumCS 18`
//!   gives `[255, 87, 0]` because the decoder has already undone it, and
//!   `EnumCS 99` gives `[255, 87, 0]` in vips because an unrecognised enum
//!   falls through to UNSPECIFIED and the subsampling turns the transform on.
//!   That third one libviprs does not read at all: `hayro-jpeg2000` refuses a
//!   `colr` box it does not recognise. It is a refusal rather than a wrong
//!   picture, and it is issue #771.
//!
//! # Divergences worth knowing about
//!
//! * **The image origin, on the size only.** A codestream may start away from
//!   the grid origin. On `origin57.j2k`, whose SIZ says `Xsiz = 37, XOsiz = 5,
//!   Ysiz = 31, YOsiz = 7`, this module reports `32x24`, which is
//!   `Xsiz - XOsiz` by `Ysiz - YOsiz` and what the standard says the image is.
//!   vips reports `27x17`, that size less the origin a second time.
//!
//!   **vips's answer is the top-left crop of this one**, measured by hashing
//!   it: our 32x24 cut to 27x17 at (0, 0) reproduces the capture's
//!   `decoded_raster.sha256` exactly, so `jp2kload` decodes 768 samples and
//!   hands back 459 of them. That is what decided #766 in this direction; the
//!   rule from #732 and #733 is to adopt vips when a difference is inside the
//!   carrier's noise and points both ways, and to keep ours when it is large
//!   or one-directional, and dropping 40% of the picture is both.
//!
//!   The **offsets** are not a divergence at all: this loader stamps
//!   `xoffset = -XOsiz`, `yoffset = -YOsiz`, which is exactly what
//!   `vipsheader` reports for the same file and exactly what `extract_area`
//!   stamps in this crate for a crop at the same place (#721). So the two
//!   agree on where the image is and disagree only on how much of it there is.
//!   Nothing `jp2ksave` writes reaches either, since it always starts at the
//!   grid origin.
//! * **The `colr` box's enumerated colour space does not decide the tag
//!   here.** `jp2kload` reads `EnumCS` and not the band count, so a
//!   one-component file tagged CMYK comes back `cmyk` and a three-component
//!   one tagged greyscale comes back `b-w`. This module takes
//!   `hayro-jpeg2000`'s colour-space answer, which agrees with vips on all 22
//!   decodable fixtures and would disagree on those two synthetic re-taggings.
//!   Filed as issue #767.

use std::path::Path;

use thiserror::Error;

use crate::codec::EncodeError;
#[cfg(feature = "jp2k")]
use crate::conversion::Interpretation;
#[cfg(feature = "jp2k")]
use crate::imageio::MetadataValue;
use crate::imageio::SaveError;
#[cfg(feature = "jp2k")]
use crate::pixel::PixelFormat;
#[cfg(feature = "jp2k")]
use crate::raster::buffer_len;
use crate::raster::{Raster, RasterError};
use crate::source::{DecodeLimits, SourceError};

/// The RFC 3745 JP2 signature box, and the first of the two signatures
/// the content sniffer in [`crate::source`] accepts for this container.
///
/// `vips_foreign_load_jp2k_get_codec_format` reads the same twelve bytes.
pub(crate) const JP2_SIGNATURE: &[u8] = b"\x00\x00\x00\x0cjP  \r\n\x87\n";

/// The `SOC` + `SIZ` marker pair opening a bare codestream, and the other
/// signature the content sniffer in [`crate::source`] accepts.
///
/// `jp2ksave` never writes this form, so every bare-codestream fixture in the
/// oracle capture had to come from `opj_compress`. The loader still has to
/// read it, because plenty of other producers write it.
pub(crate) const CODESTREAM_SIGNATURE: &[u8] = b"\xff\x4f\xff\x51";

/// The highest band count [`Raster::encode_jp2k`] will write.
///
/// Not a format limit and not an encoder limit: `jp2ksave` writes more and so
/// does `openjpeg2-pure-rs`, and measured, a five-band file written here reads
/// back through `vips jp2kload` bit for bit. The ceiling is the *loader's*.
/// `hayro-jpeg2000` refuses a component set it cannot map onto greyscale, RGB,
/// CMYK or one of those plus alpha, so anything wider is a file this crate can
/// write and cannot read. Issue #769 tracks lifting both halves.
pub const MAX_BANDS: usize = 4;

/// The highest component precision [`decode_jp2k`] will carry.
///
/// Two independent ceilings land on the same number. [`PixelFormat`] has no
/// 32-bit integer carrier, so there is nowhere to put a wider sample; and
/// `hayro-jpeg2000` returns samples as `f32`, whose 24-bit mantissa cannot
/// hold one. Measured on `int31.jp2`, whose five distinct 31-bit samples come
/// back as three distinct floats.
pub const MAX_PRECISION: u8 = 16;

/// Errors from the JPEG 2000 loader.
///
/// The enum, and the [`SourceError::Jp2k`] variant that carries it, are
/// declared whether or not the **`jp2k`** feature is on, so a caller's `match`
/// has the same arms in either build. What changes is which arms are
/// reachable: without the feature the only one is
/// [`Jp2kError::FeatureNotEnabled`], and with it that is the only one that
/// never fires.
///
/// The encoder does not report through here. [`Raster::encode_jp2k`] and
/// [`Raster::save_jp2k`] stay on the shared [`EncodeError`] spine, where
/// [`crate::jxl`], [`crate::gif`] and [`crate::fits`] leave their save
/// refusals too.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Jp2kError {
    /// The crate was built without the **`jp2k`** feature, so there is no
    /// decoder behind [`decode_jp2k`] at all.
    ///
    /// Reported instead of a missing symbol or a panic: every entry point in
    /// this module exists at the same signature in both builds. This is the
    /// only variant a build without the feature can produce, and the one
    /// variant a build *with* it never produces, which is what makes "this
    /// build has no JPEG 2000" distinguishable from "these bytes are not
    /// JPEG 2000" without reading a message.
    #[error("jp2k: JPEG 2000 decoding is not available in this build (enable the `jp2k` feature)")]
    FeatureNotEnabled,
    /// `hayro-jpeg2000` refused the bitstream: a bad signature, a malformed
    /// box, a missing codestream, a broken entropy stream.
    ///
    /// The untyped tail, for the reason [`ExrError::Decode`] is one. JPEG 2000
    /// is an ISOBMFF box reader, a marker-segment parser, an MQ arithmetic
    /// decoder, an EBCOT tier-1/tier-2 pass and two wavelet inverses, and
    /// reproducing that taxonomy here would be a second decoder. The message
    /// is the dependency's own.
    ///
    /// [`ExrError::Decode`]: crate::exr::ExrError::Decode
    #[error("jp2k: {message}")]
    Decode {
        /// The underlying decoder failure, rendered through its `Display`.
        message: String,
    },
    /// The container is structurally broken before any decoder is reached:
    /// a box length that runs off the end, a missing `jp2c`, a `SIZ` marker
    /// that is not where the codestream says it is.
    ///
    /// Separate from [`Jp2kError::Decode`] because this module parses the box
    /// structure and the `SIZ`/`COD` markers itself, for the signedness bit,
    /// the subsampling factors, the tile geometry and the `METH=2` ICC
    /// payload, none of which `hayro-jpeg2000` reports. A file can fail here
    /// and decode fine, or the reverse.
    #[error("jp2k: malformed container: {reason}")]
    Container {
        /// What was wrong, named by the structure that was being read.
        reason: String,
    },
    /// A component is signed, and there is no signed carrier to decode it
    /// into.
    ///
    /// libvips has `char` and `short` band formats; [`PixelFormat`] has
    /// neither. `hayro-jpeg2000` does not report signedness either and returns
    /// every component DC-level-shifted into the unsigned range, so decoding
    /// one anyway would come back offset by half the range with nothing
    /// saying so: measured on `depth12s.j2k`, libviprs would report 0 where
    /// vips reports -32768.
    #[error(
        "jp2k: component {component} is a signed {precision}-bit component, and a raster \
         has no signed sample carrier; vips reads it as `char` or `short`"
    )]
    SignedComponent {
        /// Which component, counting from zero, declared the sign bit.
        component: usize,
        /// The precision that component declared.
        precision: u8,
    },
    /// A component declares more bits per sample than this loader carries.
    ///
    /// The ceiling is [`MAX_PRECISION`] and the reasons are on it.
    #[error(
        "jp2k: component {component} declares {precision} bits per sample; this loader \
         carries at most {max}, because a raster has no 32-bit integer carrier and the \
         decoder returns samples as f32"
    )]
    PrecisionNotSupported {
        /// Which component, counting from zero, declared the precision.
        component: usize,
        /// The precision that component declared.
        precision: u8,
        /// The ceiling, [`MAX_PRECISION`].
        max: u8,
    },
    /// A decoded component is wider than the carrier the codestream header
    /// priced for it.
    ///
    /// Only a **palette** can make the two disagree, and it is a real file
    /// shape rather than a defensive one: a palettised codestream declares the
    /// precision of the palette *index* in `SIZ`, and the entries the `pclr`
    /// box maps it to carry their own, which can be wider. Measured on a
    /// hand-built JP2 with an 8-bit index and 16-bit palette columns, this is
    /// what libviprs says and `vips jp2kload` fails the same file with
    /// `error in tile 0`, so the refusal is not a divergence.
    ///
    /// Separate from [`Jp2kError::PrecisionNotSupported`] because that one is
    /// about [`MAX_PRECISION`], which this has not reached: the decoded
    /// component here can be well inside the ceiling and still be wider than
    /// the element the `SIZ` scan chose and the allocation budget was spent
    /// on. Reporting it as the ceiling would name a number that is not the
    /// one in the way.
    #[error(
        "jp2k: component {component} decoded to {decoded} bits per sample where the \
         codestream header declared at most {declared}, so the sample carrier was \
         chosen and the frame priced for a narrower component (a palette is the only \
         thing that can widen one)"
    )]
    PrecisionWiderThanDeclared {
        /// Which component, counting from zero.
        component: usize,
        /// The widest precision the `SIZ` marker declared, which is what
        /// picked the carrier.
        declared: u8,
        /// What the decoder returned for this component.
        decoded: u8,
    },
    /// The band count has no [`PixelFormat`] carrier.
    ///
    /// Defensive: a raster holds 1 to `u16::MAX` bands and a JPEG 2000
    /// codestream declares at most 16384 components, so nothing a valid file
    /// can say reaches this. It exists because the carrier is chosen from a
    /// number rather than from an enum, and a band count of zero would
    /// otherwise be a zero-sized buffer.
    #[error(
        "jp2k: a JPEG 2000 with {bands} bands has no raster carrier; a raster holds 1 to {max} bands"
    )]
    UnsupportedBandCount {
        /// The band count the header implied.
        bands: u32,
        /// The ceiling, `u16::MAX`.
        max: u32,
    },
    /// The decoded component set does not match what the header declared.
    ///
    /// Defensive, and load-bearing: the header's count is what the allocation
    /// budget below was spent on and what the output buffer is sized for, so a
    /// decode that disagrees would be written into a buffer sized for
    /// something else.
    #[error("jp2k: the header declared {declared} bands and the decode produced {decoded}")]
    BandCountMismatch {
        /// The band count the image header declared.
        declared: u32,
        /// The component count the decode produced.
        decoded: u32,
    },
    /// A decoded component carries a different number of samples than the
    /// declared geometry implies.
    ///
    /// Defensive, for the same reason as [`Jp2kError::BandCountMismatch`]:
    /// `hayro-jpeg2000` upsamples a subsampled component to the full grid
    /// itself, so every component should come back at `width * height`, and a
    /// component that does not would be read past its end.
    #[error(
        "jp2k: component {component} decoded to {decoded} samples where the {width}x{height} \
         geometry needs {expected}"
    )]
    ComponentGeometryMismatch {
        /// Which component, counting from zero.
        component: usize,
        /// The declared width.
        width: u32,
        /// The declared height.
        height: u32,
        /// `width * height`.
        expected: usize,
        /// What the component actually carried.
        decoded: usize,
    },
    /// Constructing the decoded [`Raster`] failed.
    #[error(transparent)]
    Raster(#[from] RasterError),
}

/// How the JPEG 2000 encoder compresses pixels (libvips `jp2ksave`'s
/// `lossless` flag and its `Q`, as far as either is reachable here).
///
/// `#[non_exhaustive]` so a `Q`-shaped mode can join it as a minor bump if the
/// encoder ever exposes `cp_fixed_quality`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Compression {
    /// Reversible 5/3 wavelet with no quantisation (libvips
    /// `jp2ksave --lossless`). The pixels round-trip exactly.
    #[default]
    Lossless,
    /// Irreversible 9/7 wavelet at a target compression ratio.
    ///
    /// `ratio` is OpenJPEG's `tcp_rates` under `cp_disto_alloc`: a value of
    /// 20 asks for a file about twenty times smaller than the raw samples.
    /// It is **not** `jp2ksave`'s `Q`, which is a distortion ratio in
    /// decibels set through `cp_fixed_quality`; that knob is `pub(crate)` in
    /// `openjpeg2-pure-rs` and so is not reachable from here at all. Naming
    /// the field `ratio` rather than `Q` is what keeps this module from
    /// accepting a number and quietly meaning something else by it.
    Lossy {
        /// The target compression ratio, at least 1.
        ratio: std::num::NonZeroU16,
    },
}

/// Options for [`Raster::encode_jp2k`] (libvips `jp2ksave` /
/// `jp2ksave_buffer`).
///
/// `#[non_exhaustive]`, `Default`, and module-scoped, the same shape as
/// [`DecodeLimits`]: start from [`SaveOptions::default`] and set what you need
/// with the `with_*` builders, e.g.
/// `jp2k::SaveOptions::default().with_compression(compression)`. That is what
/// makes "later fields can be added without a breaking change" true rather
/// than merely written down (issue #630).
///
/// There is no `keep` field and no `profile` field, because `jp2ksave` has
/// neither behind it: it inherits both from `VipsForeignSave` and implements
/// neither, and saving with a profile produces a byte-identical file
/// (measured). There is no tile geometry and no `subsample_mode` either;
/// `openjpeg2-pure-rs` exposes `format`, `threads`, `irreversible`,
/// `use_mct`, `rates` and `num_resolutions` and keeps the rest of
/// `opj_cparameters_t` `pub(crate)`, so neither has an encoder behind it.
/// Tiled save is issue #768.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct SaveOptions {
    /// How to compress. Defaults to [`Compression::Lossless`].
    pub compression: Compression,
}

impl SaveOptions {
    /// Set the compression mode, returning the updated options.
    #[must_use]
    pub fn with_compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }
}

/// Decode JPEG 2000 bytes into a [`Raster`] (libvips `jp2kload_buffer` at its
/// default `page = 0`).
///
/// Both container forms decode: the RFC 3745 JP2 box structure, which starts
/// with the twelve-byte signature box, and the bare codestream, which starts
/// `FF 4F FF 51`. The name is never consulted, the way `jp2kload` never
/// consults it.
///
/// The sample carrier follows the file: a component precision of 8 or under
/// gives the 8-bit formats and 9 to 16 gives the 16-bit ones, and the sample
/// is left-justified into the element the way `vips_foreign_load_jp2k_ljust`
/// does it, so the real depth survives in `bits-per-sample` rather than in the
/// value. The band count is the colour channels plus alpha, so a greyscale
/// file stays one band.
///
/// the image origin as a negative `xoffset` / `yoffset`,
/// `icc-profile-data`, `bits-per-sample`, `jp2k-resolutions` and, when the
/// image is more than one tile, `tile-width` and `tile-height` are lifted onto
/// the raster. The first two use the names `jp2kload` uses; the third does not,
/// because vips calls it `n-pages` and this crate reserves that key for counts
/// a zero-based `page` argument can select (issue #635). See the module docs.
///
/// # Errors
///
/// Every JPEG 2000 refusal arrives as [`SourceError::Jp2k`] wrapping a
/// [`Jp2kError`]; the three ceilings below are the shared ones every codec in
/// the crate reports the same way.
///
/// * [`Jp2kError::FeatureNotEnabled`] when the crate was built without the
///   `jp2k` feature. Every bullet below needs the feature to be reachable at
///   all.
/// * [`Jp2kError::Container`] for a box structure or marker segment this
///   module cannot walk, and [`Jp2kError::Decode`] for a bitstream
///   `hayro-jpeg2000` refuses.
/// * [`Jp2kError::SignedComponent`] for a signed component and
///   [`Jp2kError::PrecisionNotSupported`] for one above [`MAX_PRECISION`],
///   both of which are carrier gaps rather than format ones, and
///   [`Jp2kError::PrecisionWiderThanDeclared`] for a palette whose entries are
///   wider than the index `SIZ` declared.
/// * [`Jp2kError::UnsupportedBandCount`],
///   [`Jp2kError::BandCountMismatch`] and
///   [`Jp2kError::ComponentGeometryMismatch`], all defensive.
/// * [`Jp2kError::Raster`] when the decoded frame cannot be wrapped.
/// * [`SourceError::CoordLimitExceeded`] when either declared axis exceeds
///   [`DecodeLimits::max_coord`], [`SourceError::DimensionLimitExceeded`] when
///   `width * height` exceeds `max_pixels`, and
///   [`SourceError::AllocLimitExceeded`] when the component buffers the header
///   declares would exceed `max_alloc_bytes`.
pub fn decode_jp2k(bytes: &[u8], limits: DecodeLimits) -> Result<Raster, SourceError> {
    decode(bytes, limits)
}

/// The `jp2k`-feature-off body of [`decode_jp2k`]: the one [`Jp2kError`]
/// variant this build can produce, so a caller compiled either way sees one
/// signature and one error type and can tell "this build has no JPEG 2000"
/// from "these bytes are not JPEG 2000" by the variant rather than by the
/// message.
#[cfg(not(feature = "jp2k"))]
fn decode(bytes: &[u8], limits: DecodeLimits) -> Result<Raster, SourceError> {
    let _ = (bytes, limits);
    Err(Jp2kError::FeatureNotEnabled.into())
}

impl Raster {
    /// Encode as JPEG 2000 bytes in a JP2 container (libvips
    /// `jp2ksave_buffer`).
    ///
    /// Accepts the 8- and 16-bit carriers at one to [`MAX_BANDS`] bands:
    /// [`PixelFormat::Gray8`], [`PixelFormat::Gray16`],
    /// [`PixelFormat::Rgb8`], [`PixelFormat::Rgb16`],
    /// [`PixelFormat::Rgba8`], [`PixelFormat::Rgba16`] and the two-band
    /// [`PixelFormat::Multi8`] / [`PixelFormat::Multi16`] forms. The float
    /// carriers are refused rather than cast, which is what `vips jp2ksave`
    /// does with a `float` or `double` image, and so is anything wider than
    /// [`MAX_BANDS`], for the reason on that constant: `jp2ksave` writes it
    /// and [`decode_jp2k`] cannot read it back.
    ///
    /// The output is always a JP2 container, never a bare codestream, because
    /// `jp2ksave` hard-codes `OPJ_CODEC_JP2` and writes the same bytes for all
    /// five suffixes it registers. Nothing attached to the raster is written:
    /// no ICC profile, no EXIF, no XMP, because `jp2ksave.c` has no code for
    /// any of them.
    ///
    /// # Errors
    ///
    /// [`EncodeError::Unsupported`] naming `"jp2k"` when the crate was built
    /// without the `jp2k` feature, which is the variant every format without
    /// an encoder in this build reports; otherwise [`EncodeError::Encode`]
    /// when the raster is float (cast first; the message says so), when it
    /// has more than [`MAX_BANDS`] bands, or when the codec rejects the
    /// frame.
    pub fn encode_jp2k(&self, options: SaveOptions) -> Result<Vec<u8>, EncodeError> {
        encode(self, options)
    }

    /// Save the raster to `path` as JPEG 2000 (libvips `jp2ksave`).
    ///
    /// The bytes are a JP2 container whatever `path` is called, for the reason
    /// on [`Raster::encode_jp2k`].
    ///
    /// # Errors
    ///
    /// [`SaveError::Encode`] when [`Raster::encode_jp2k`] rejects the raster,
    /// or [`SaveError::Io`] when the file write fails.
    pub fn save_jp2k(&self, path: &Path, options: SaveOptions) -> Result<(), SaveError> {
        let bytes = self.encode_jp2k(options).map_err(encode_to_save)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }
}

/// The `jp2k`-feature-off body of [`Raster::encode_jp2k`].
#[cfg(not(feature = "jp2k"))]
fn encode(raster: &Raster, options: SaveOptions) -> Result<Vec<u8>, EncodeError> {
    let _ = (raster, options);
    Err(EncodeError::unsupported("jp2k"))
}

/// The extension route's entry point (`imageio.rs`'s `.jp2` / `.j2k` / `.jpt`
/// / `.j2c` / `.jpc` arm), at the `jp2ksave` defaults.
///
/// Takes no `keep_metadata`, for the same reason [`crate::jxl`]'s twin does
/// not: there is nothing to drop. `jp2ksave.c` writes no ICC profile, no EXIF
/// block and no XMP packet, so a stripped save and a kept one produce the same
/// bytes and a flag here would be a promise with nothing behind it.
#[cfg(feature = "jp2k")]
pub(crate) fn encode_jp2k_for_save(raster: &Raster) -> Result<Vec<u8>, SaveError> {
    raster
        .encode_jp2k(SaveOptions::default())
        .map_err(encode_to_save)
}

/// Carry an [`EncodeError`] onto the save spine, the way [`crate::jxl`] does:
/// an I/O failure stays an I/O failure and everything else flattens onto the
/// sink's message variant, which is the only shape [`SaveError::Encode`] has.
fn encode_to_save(err: EncodeError) -> SaveError {
    match err {
        EncodeError::Io(io) => SaveError::Io(io),
        other => SaveError::Encode(crate::sink::SinkError::EncodeMsg(other.to_string())),
    }
}

// ---------------------------------------------------------------------------
// The container and codestream structure, which this module reads itself
// ---------------------------------------------------------------------------

/// Where the contiguous codestream starts, and how it was wrapped.
///
/// `hayro-jpeg2000` reads the boxes too and reports none of what is in them
/// beyond a colour space it could parse, so this module walks them again for
/// four things it needs and cannot get otherwise: whether the file is a bare
/// codestream (which is what makes the colour space unspecified, which is what
/// turns the inverse YCC on), the `METH=2` ICC payload (which the decoder
/// drops when it cannot parse it, and `icc_colr.jp2` deliberately cannot be
/// parsed), the per-component sign bit and subsampling factors, and the tile
/// geometry.
#[cfg(feature = "jp2k")]
#[derive(Clone, Debug, PartialEq, Eq)]
struct ContainerLayout {
    /// Offset of the `SOC` marker opening the contiguous codestream.
    codestream: usize,
    /// True for a bare codestream, false for a JP2 box structure.
    ///
    /// This is the whole of the "unspecified colour space" test for the files
    /// that reach it: a JP2 always carries a `colr` box, so a bare codestream
    /// is the only shape whose colour space is unspecified by construction.
    bare: bool,
    /// The payload of a `METH=2` `colr` box, which is an ICC profile, copied
    /// verbatim and unvalidated the way `jp2kload` copies it.
    icc: Option<Vec<u8>>,
}

/// One top-level or sub-box, as an offset range into the file.
#[cfg(feature = "jp2k")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BoxRef {
    kind: [u8; 4],
    /// First byte of the payload.
    start: usize,
    /// One past the last byte of the payload.
    end: usize,
}

/// A short reason string, wrapped as [`Jp2kError::Container`].
#[cfg(feature = "jp2k")]
fn container(reason: impl Into<String>) -> Jp2kError {
    Jp2kError::Container {
        reason: reason.into(),
    }
}

/// Read a big-endian `u16` at `at`, or report what ran off the end.
#[cfg(feature = "jp2k")]
fn be16(bytes: &[u8], at: usize, what: &str) -> Result<u16, Jp2kError> {
    let s = bytes
        .get(at..at + 2)
        .ok_or_else(|| container(format!("{what} runs past the end of the file")))?;
    Ok(u16::from_be_bytes([s[0], s[1]]))
}

/// Read a big-endian `u32` at `at`, or report what ran off the end.
#[cfg(feature = "jp2k")]
fn be32(bytes: &[u8], at: usize, what: &str) -> Result<u32, Jp2kError> {
    let s = bytes
        .get(at..at + 4)
        .ok_or_else(|| container(format!("{what} runs past the end of the file")))?;
    Ok(u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
}

/// Walk the boxes in `bytes[from..to]`, returning each one's kind and payload
/// range.
///
/// ISO/IEC 15444-1 Annex I: every box is a 4-byte big-endian `LBox`, a 4-byte
/// `TBox`, and then either the payload or, when `LBox` is 1, an 8-byte `XLBox`
/// before it. `LBox = 0` means "to the end of the enclosing range", which is
/// legal only for the last box.
///
/// Termination is by construction rather than by a counter: every accepted box
/// advances `at` by at least 8, and a box that would not is the error below.
#[cfg(feature = "jp2k")]
fn walk_boxes(bytes: &[u8], from: usize, to: usize) -> Result<Vec<BoxRef>, Jp2kError> {
    let mut out = Vec::new();
    let mut at = from;
    while at + 8 <= to {
        let lbox = be32(bytes, at, "a box length")?;
        let kind: [u8; 4] = bytes[at + 4..at + 8]
            .try_into()
            .expect("four bytes are four bytes");
        let (header, total) = match lbox {
            0 => (8usize, to - at),
            1 => {
                let s = bytes
                    .get(at + 8..at + 16)
                    .ok_or_else(|| container("an extended box length runs past the end"))?;
                let xl = u64::from_be_bytes(s.try_into().expect("eight bytes are eight bytes"));
                let xl = usize::try_from(xl)
                    .map_err(|_| container("an extended box length does not fit in memory"))?;
                (16usize, xl)
            }
            n if n >= 8 => (8usize, n as usize),
            n => {
                return Err(container(format!(
                    "a box declares a length of {n}, which is shorter than its own header"
                )));
            }
        };
        if total < header {
            return Err(container(format!(
                "a box declares a length of {total}, which is shorter than its {header}-byte header"
            )));
        }
        let end = at
            .checked_add(total)
            .ok_or_else(|| container("a box length overflows the address space"))?;
        if end > to {
            return Err(container(format!(
                "a {} box runs {} bytes past the end of the file",
                String::from_utf8_lossy(&kind),
                end - to
            )));
        }
        out.push(BoxRef {
            kind,
            start: at + header,
            end,
        });
        at = end;
    }
    Ok(out)
}

#[cfg(feature = "jp2k")]
impl ContainerLayout {
    /// Identify the container and pull out of it the two things the decoder
    /// does not report: where the codestream is, and the raw ICC profile.
    fn parse(bytes: &[u8]) -> Result<Self, Jp2kError> {
        if bytes.starts_with(JP2_SIGNATURE) {
            let top = walk_boxes(bytes, 0, bytes.len())?;
            let codestream = top
                .iter()
                .find(|b| &b.kind == b"jp2c")
                .map(|b| b.start)
                .ok_or_else(|| container("no jp2c box, so the file carries no codestream"))?;
            let mut icc = None;
            for header in top.iter().filter(|b| &b.kind == b"jp2h") {
                icc = walk_boxes(bytes, header.start, header.end)?
                    .into_iter()
                    .filter(|b| &b.kind == b"colr")
                    .find_map(|b| icc_payload(&bytes[b.start..b.end]));
                if icc.is_some() {
                    break;
                }
            }
            Ok(Self {
                codestream,
                bare: false,
                icc,
            })
        } else if bytes.starts_with(CODESTREAM_SIGNATURE) {
            Ok(Self {
                codestream: 0,
                bare: true,
                icc: None,
            })
        } else {
            Err(container(
                "the leading bytes are neither the JP2 signature box nor a SOC + SIZ pair",
            ))
        }
    }
}

/// The ICC profile inside a `colr` box, or `None` for any other method.
///
/// `colr` is `METH` (1), `PREC` (1), `APPROX` (1), then either a 4-byte
/// `EnumCS` when `METH` is 1 or the profile bytes when it is 2. The payload is
/// taken verbatim and is not validated, which is what `jp2kload` does and is
/// exactly why the `icc_colr.jp2` fixture (24 bytes that are not an ICC
/// profile at all) has a profile in vips and does not in `hayro-jpeg2000`.
#[cfg(feature = "jp2k")]
fn icc_payload(payload: &[u8]) -> Option<Vec<u8>> {
    match payload.first() {
        Some(2) if payload.len() > 3 => Some(payload[3..].to_vec()),
        _ => None,
    }
}

/// What one component's `Ssiz` / `XRsiz` / `YRsiz` triple declares.
#[cfg(feature = "jp2k")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ComponentSpec {
    /// Bits per sample, which is `(Ssiz & 0x7f) + 1`.
    precision: u8,
    /// The `Ssiz` sign bit.
    signed: bool,
    /// Horizontal subsampling factor, `XRsiz`.
    dx: u8,
    /// Vertical subsampling factor, `YRsiz`.
    dy: u8,
}

/// What the codestream's own `SIZ` and `COD` markers declare.
#[cfg(feature = "jp2k")]
#[derive(Clone, Debug, PartialEq, Eq)]
struct CodestreamHeader {
    /// `Xsiz`, the reference grid width including the image origin.
    xsiz: u32,
    /// `Ysiz`.
    ysiz: u32,
    /// `XOsiz`, the image origin on the reference grid.
    x_origin: u32,
    /// `YOsiz`.
    y_origin: u32,
    /// `XTsiz`, the tile width on the reference grid.
    tile_width: u32,
    /// `YTsiz`.
    tile_height: u32,
    /// `XTOsiz`, the tile grid origin.
    tile_x_origin: u32,
    /// `YTOsiz`.
    tile_y_origin: u32,
    /// One entry per component, in codestream order.
    components: Vec<ComponentSpec>,
    /// `SPcod`'s decomposition level count plus one, or `None` when no `COD`
    /// marker was found before the first tile.
    ///
    /// This is what `vipsheader` reports as `n-pages` and what vips's `page`
    /// argument indexes; see the module docs for why it does not travel under
    /// that key here.
    resolutions: Option<u16>,
}

/// The `SOC` marker.
#[cfg(feature = "jp2k")]
const MARKER_SOC: u16 = 0xff4f;
/// The `SIZ` marker, the image and tile size segment.
#[cfg(feature = "jp2k")]
const MARKER_SIZ: u16 = 0xff51;
/// The `COD` marker, the coding style default segment.
#[cfg(feature = "jp2k")]
const MARKER_COD: u16 = 0xff52;
/// `SOT`, the first tile-part header, which ends the main header.
#[cfg(feature = "jp2k")]
const MARKER_SOT: u16 = 0xff90;
/// `SOD`, the start of data.
#[cfg(feature = "jp2k")]
const MARKER_SOD: u16 = 0xff93;
/// `EOC`, the end of the codestream.
#[cfg(feature = "jp2k")]
const MARKER_EOC: u16 = 0xffd9;

#[cfg(feature = "jp2k")]
impl CodestreamHeader {
    /// Parse the main header of the codestream starting at `bytes[0]`.
    ///
    /// Only `SIZ` and `COD` are read; every other segment is skipped by its
    /// own length. The walk stops at the first `SOT`, which is where the main
    /// header ends, so no tile data is touched.
    fn parse(bytes: &[u8]) -> Result<Self, Jp2kError> {
        if be16(bytes, 0, "the SOC marker")? != MARKER_SOC {
            return Err(container("the codestream does not open with a SOC marker"));
        }
        if be16(bytes, 2, "the SIZ marker")? != MARKER_SIZ {
            return Err(container("no SIZ marker follows SOC"));
        }
        let lsiz = usize::from(be16(bytes, 4, "the SIZ segment length")?);
        let count = usize::from(be16(bytes, 40, "the SIZ component count")?);
        if count == 0 {
            return Err(container("the SIZ marker declares no components"));
        }
        let mut components = Vec::with_capacity(count.min(1 << 14));
        for i in 0..count {
            let at = 42 + i * 3;
            let ssiz = *bytes
                .get(at)
                .ok_or_else(|| container(format!("the Ssiz byte for component {i} is missing")))?;
            components.push(ComponentSpec {
                precision: (ssiz & 0x7f) + 1,
                signed: ssiz & 0x80 != 0,
                dx: bytes[at + 1],
                dy: bytes[at + 2],
            });
        }
        if components.iter().any(|c| c.dx == 0 || c.dy == 0) {
            return Err(container(
                "a component declares a zero subsampling factor, which would divide by zero",
            ));
        }

        let mut header = Self {
            xsiz: be32(bytes, 8, "Xsiz")?,
            ysiz: be32(bytes, 12, "Ysiz")?,
            x_origin: be32(bytes, 16, "XOsiz")?,
            y_origin: be32(bytes, 20, "YOsiz")?,
            tile_width: be32(bytes, 24, "XTsiz")?,
            tile_height: be32(bytes, 28, "YTsiz")?,
            tile_x_origin: be32(bytes, 32, "XTOsiz")?,
            tile_y_origin: be32(bytes, 36, "YTOsiz")?,
            components,
            resolutions: None,
        };

        // Every marker segment after SIZ is two marker bytes and a length that
        // counts itself but not the marker, so `2 + length` is the stride. The
        // walk terminates because that stride is at least 4 and the three
        // delimiters below carry no length at all.
        let mut at = 4 + lsiz;
        while at + 2 <= bytes.len() {
            let marker = be16(bytes, at, "a marker")?;
            if marker == MARKER_SOT || marker == MARKER_SOD || marker == MARKER_EOC {
                break;
            }
            if marker >> 8 != 0xff {
                return Err(container(format!(
                    "expected a marker in the main header and found {marker:#06x}"
                )));
            }
            let length = usize::from(be16(bytes, at + 2, "a marker segment length")?);
            if length < 2 {
                return Err(container(format!(
                    "a {marker:#06x} segment declares a length of {length}, which is shorter \
                     than the length field itself"
                )));
            }
            if marker == MARKER_COD {
                // Scod (1), then SGcod: progression order (1), layer count
                // (2), multiple-component transform (1), then SPcod, whose
                // first byte is the decomposition level count.
                let levels = *bytes.get(at + 9).ok_or_else(|| {
                    container("the COD segment ends before its decomposition level count")
                })?;
                header.resolutions = Some(u16::from(levels) + 1);
                break;
            }
            at += 2 + length;
        }
        Ok(header)
    }

    /// The image origin as the offset vips records for it: `-XOsiz` by
    /// `-YOsiz`.
    ///
    /// Negative, because that is what `jp2kload` writes (`xoffset = -5` for
    /// `origin57.j2k`, measured) and because it is this crate's own
    /// convention for the same meaning: `extract_area` stamps `-left` /
    /// `-top` (#721), and a codestream whose image region begins at `x = 5` on
    /// the reference grid is exactly that crop of the grid.
    ///
    /// So the loader and vips agree on the offsets even where they disagree on
    /// the size, which is the whole of the remaining #766 divergence.
    fn origin_offset(&self) -> (i32, i32) {
        (negated_origin(self.x_origin), negated_origin(self.y_origin))
    }

    /// The image size, which is the reference grid less the image origin.
    ///
    /// This is `Xsiz - XOsiz` by `Ysiz - YOsiz`, which is what the standard
    /// says the image is and what `hayro-jpeg2000` reports. vips subtracts the
    /// origin a second time and hands back the top-left crop of this, measured
    /// by digest; see the module docs and issue #766. Where the image sits is
    /// reported separately, by [`CodestreamHeader::origin_offset`].
    fn image_size(&self) -> (u32, u32) {
        (
            self.xsiz.saturating_sub(self.x_origin),
            self.ysiz.saturating_sub(self.y_origin),
        )
    }

    /// How many tiles across and down the codestream is cut into.
    ///
    /// `vipsheader` attaches `tile-width` and `tile-height` only when this is
    /// more than one tile, so a port that always attaches them disagrees with
    /// vips on every small image.
    fn tile_grid(&self) -> (u32, u32) {
        let across = tile_count(self.xsiz, self.tile_x_origin, self.tile_width);
        let down = tile_count(self.ysiz, self.tile_y_origin, self.tile_height);
        (across, down)
    }

    /// The highest precision any component declares, which is what picks the
    /// sample carrier.
    fn max_precision(&self) -> u8 {
        self.components
            .iter()
            .map(|c| c.precision)
            .max()
            .unwrap_or(8)
    }

    /// Whether this is the shape `jp2kload` reads as YCC: three components
    /// with the second and third subsampled against the first.
    ///
    /// The other half of vips's condition, an unspecified colour space, is
    /// [`ContainerLayout::bare`] and is checked by the caller.
    fn chroma_subsampled(&self) -> bool {
        self.components.len() == 3
            && self.components[0].dx == 1
            && self.components[0].dy == 1
            && self.components[1..].iter().all(|c| c.dx > 1 || c.dy > 1)
    }
}

/// Tiles along one axis: `ceil((size - origin) / tile)`, and 1 for a tile size
/// of zero, which a malformed header can declare.
#[cfg(feature = "jp2k")]
fn tile_count(size: u32, origin: u32, tile: u32) -> u32 {
    if tile == 0 {
        return 1;
    }
    let span = size.saturating_sub(origin);
    span.div_ceil(tile).max(1)
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

/// The `jp2k`-feature-on body of [`decode_jp2k`].
#[cfg(feature = "jp2k")]
fn decode(bytes: &[u8], limits: DecodeLimits) -> Result<Raster, SourceError> {
    use hayro_jpeg2000::{DecodeSettings, DecoderContext, Image as Jp2kImage};

    // Pass one over the bytes: the box structure and the main header, for the
    // four things the decoder does not report. Both are bounded reads over
    // slices that are already in memory, so neither costs an allocation the
    // budget below would want to know about.
    let layout = ContainerLayout::parse(bytes)?;
    let header = CodestreamHeader::parse(&bytes[layout.codestream..])?;

    // The carrier gaps, answered off the codestream's own SIZ rather than off
    // the decoder, because `hayro-jpeg2000` reports neither the sign bit nor a
    // per-component precision before the decode, and both decide whether the
    // file can be carried at all.
    for (component, spec) in header.components.iter().enumerate() {
        if spec.signed {
            return Err(Jp2kError::SignedComponent {
                component,
                precision: spec.precision,
            }
            .into());
        }
        if spec.precision > MAX_PRECISION {
            return Err(Jp2kError::PrecisionNotSupported {
                component,
                precision: spec.precision,
                max: MAX_PRECISION,
            }
            .into());
        }
    }

    let image = Jp2kImage::new(bytes, &DecodeSettings::default()).map_err(decode_error)?;
    let (width, height) = (image.width(), image.height());
    // Two parsers read the same `SIZ`, and the allocation budget below is
    // spent on this one's answer while the sign bit and the subsampling
    // factors come from the other's. A file they read differently is a file
    // where one of those two decisions is being made about a geometry the
    // other did not see, so it is refused rather than reconciled.
    if header.image_size() != (width, height) {
        let (xsiz, ysiz) = header.image_size();
        return Err(container(format!(
            "the SIZ marker declares a {xsiz}x{ysiz} image and the decoder reports \
             {width}x{height}"
        ))
        .into());
    }
    limits.check_coord(width, height)?;
    limits.check_pixels(width, height)?;

    // The band count is the colour channels plus alpha, after any palette has
    // been resolved, which is why it comes off the decoder and not off `SIZ`:
    // a palettised codestream declares one component and decodes to three.
    let bands = u32::from(image.color_space().num_channels()) + u32::from(image.has_alpha());
    let element_bytes: u64 = if header.max_precision() > 8 { 2 } else { 1 };
    let format = carrier(bands, element_bytes)?;

    // The allocation budget, which `check_pixels` does not imply: a
    // 1-gigapixel `max_pixels` permits an 8 GiB `Rgba16` frame against a
    // 512 MiB default budget. Priced from the declared geometry before the
    // decoder reserves anything, the way `crate::exr` and `crate::fits` price
    // theirs, and reported through the one shared shape (issue #686).
    limits.check_image_alloc(
        "JPEG 2000 component buffers",
        width,
        height,
        u64::from(bands),
        element_bytes,
    )?;
    // Clearing the budget is not what makes this safe to compute: on a 32-bit
    // target a caller who lifts `max_alloc_bytes` can still pass a geometry
    // whose sample count pins a `usize`. `buffer_len` does the same widened
    // product and checks the narrowing, so the count is right by construction.
    let samples = buffer_len(width, height, bands as usize).map_err(Jp2kError::Raster)?;

    // Pass two: the actual decode.
    let mut context = DecoderContext::default();
    let decoded = image.decode(&mut context).map_err(decode_error)?;
    let components = decoded.components();

    let decoded_bands = u32::try_from(components.len()).unwrap_or(u32::MAX);
    if decoded_bands != bands {
        return Err(Jp2kError::BandCountMismatch {
            declared: bands,
            decoded: decoded_bands,
        }
        .into());
    }
    let plane = width as usize * height as usize;
    let element_bits = element_bytes as u32 * 8;
    for (component, data) in components.iter().enumerate() {
        if data.samples().len() != plane {
            return Err(Jp2kError::ComponentGeometryMismatch {
                component,
                width,
                height,
                expected: plane,
                decoded: data.samples().len(),
            }
            .into());
        }
        // The shift below is `element_bits - bit_depth`, so a decoded
        // component wider than the carrier the `SIZ` scan priced would
        // underflow it and then shift by more bits than the element has. The
        // loop over `SIZ` above is what keeps the two in step for every file
        // whose components are what the header says; this is the second half,
        // for the one thing `SIZ` cannot predict, which is what a palette
        // expands to. It reports its own variant because the number in the way
        // is the declared width and not `MAX_PRECISION`.
        if u32::from(data.bit_depth()) > element_bits {
            return Err(Jp2kError::PrecisionWiderThanDeclared {
                component,
                declared: header.max_precision(),
                decoded: data.bit_depth(),
            }
            .into());
        }
    }

    let mut buffer = vec![0u8; samples * element_bytes as usize];
    let ycc = layout.bare && header.chroma_subsampled();
    if ycc {
        // `jp2kload` runs OpenJPEG's `sycc_to_rgb` over the component values
        // at their own precision, before the left-justification, so this does
        // too. Component 0's precision is the one the transform is centred on,
        // which is what `offset` and `upb` below come from.
        let precision = u32::from(components[0].bit_depth());
        let shifts: Vec<u32> = components
            .iter()
            .map(|c| element_bits - u32::from(c.bit_depth()))
            .collect();
        let (y_plane, cb_plane, cr_plane) = (
            components[0].samples(),
            components[1].samples(),
            components[2].samples(),
        );
        for i in 0..plane {
            let y = quantise(y_plane[i], precision);
            let cb = quantise(cb_plane[i], precision);
            let cr = quantise(cr_plane[i], precision);
            for (band, value) in sycc_to_rgb(y, cb, cr, precision).into_iter().enumerate() {
                store(
                    &mut buffer,
                    (i * bands as usize) + band,
                    element_bytes,
                    value << shifts[band],
                );
            }
        }
    } else {
        for (band, data) in components.iter().enumerate() {
            let precision = u32::from(data.bit_depth());
            let shift = element_bits - precision;
            for (i, sample) in data.samples().iter().enumerate() {
                let value = quantise(*sample, precision) << shift;
                store(
                    &mut buffer,
                    (i * bands as usize) + band,
                    element_bytes,
                    value,
                );
            }
        }
    }

    let mut raster = Raster::new(width, height, format, buffer).map_err(Jp2kError::Raster)?;
    raster.meta.interpretation = Some(interpretation(image.color_space(), element_bytes));
    // Where the image sits on the reference grid, as the negative offset
    // `jp2kload` records and `extract_area` stamps for the same meaning
    // (#766, #721). Zero for everything `jp2ksave` writes, since it always
    // starts at the grid origin.
    let (xoffset, yoffset) = header.origin_offset();
    raster.meta.xoffset = xoffset;
    raster.meta.yoffset = yoffset;
    if let Some(icc) = layout.icc {
        raster
            .fields
            .set("icc-profile-data", MetadataValue::Blob(icc));
    }
    raster.fields.set(
        "bits-per-sample",
        MetadataValue::Int(i64::from(header.components[0].precision)),
    );
    if let Some(resolutions) = header.resolutions {
        raster.fields.set(
            "jp2k-resolutions",
            MetadataValue::Int(i64::from(resolutions)),
        );
    }
    let (across, down) = header.tile_grid();
    if across > 1 || down > 1 {
        raster.fields.set(
            "tile-width",
            MetadataValue::Int(i64::from(header.tile_width)),
        );
        raster.fields.set(
            "tile-height",
            MetadataValue::Int(i64::from(header.tile_height)),
        );
    }
    Ok(raster)
}

/// Negate an origin read off `SIZ`, saturating instead of wrapping.
///
/// `XOsiz` is a 32-bit field an attacker writes, and `-XOsiz` does not fit an
/// `i32` above `2^31`. Going through `i64` first makes `2^31` negate exactly
/// onto `i32::MIN` and everything above it saturate there, which is the one
/// boundary worth getting right: `src/extract.rs`'s twin of this helper
/// shipped saturating at `i32::MIN + 1` and survived review because the fix
/// was "asserted only by reasoning" (#706). It is a private function of a
/// `u32`, so reasoning was never necessary.
///
/// Deliberately a second copy rather than a shared helper: the twin is
/// private to `crate::extract` and that module belongs to another lane. Six
/// lines and a test beat a cross-module edit here.
#[cfg(feature = "jp2k")]
fn negated_origin(v: u32) -> i32 {
    i32::try_from(-i64::from(v)).unwrap_or(i32::MIN)
}

/// Round a reconstructed sample onto the integer grid its precision defines.
///
/// The irreversible 9/7 inverse is float-specified and its output overshoots
/// both ends of the range (measured on `chroma_sub_off.jp2`: -0.61 and 255.61
/// on an 8-bit component), so the clamp is not defensive, it is the format.
///
/// Rounding rather than truncating is measured rather than assumed: against
/// the four lossy fixtures it is never worse than truncation and is better on
/// two of them, taking `chroma_sub_off`'s worst disagreement with vips from 3
/// counts to 2.
#[cfg(feature = "jp2k")]
fn quantise(sample: f32, precision: u32) -> u32 {
    let max = (1u32 << precision) - 1;
    sample.round().clamp(0.0, max as f32) as u32
}

/// Write one sample into the interleaved output buffer, native-endian for the
/// 16-bit carrier the way every other decoder in the crate writes one.
#[cfg(feature = "jp2k")]
fn store(buffer: &mut [u8], index: usize, element_bytes: u64, value: u32) {
    if element_bytes == 1 {
        buffer[index] = value as u8;
    } else {
        let at = index * 2;
        buffer[at..at + 2].copy_from_slice(&(value as u16).to_ne_bytes());
    }
}

/// OpenJPEG's inverse YCC, coefficient for coefficient and cast for cast.
///
/// This is `sycc_to_rgb` from OpenJPEG's `color.c`, which `jp2kload` reaches
/// through `vips_foreign_load_jp2k_get_ycc`. The three details that matter are
/// all things a textbook BT.601 inverse gets differently: the coefficients are
/// 1.402, 0.344, 0.714 and 1.772 rather than the exact 0.344136 / 0.714136;
/// the green term is one cast over the *sum* rather than two casts over the
/// halves; and every cast truncates toward zero rather than rounding.
///
/// Measured on `sub420.j2k`: this returns `[242, 98, 0]` at pixel 2 where a
/// rounding implementation returns `[243, 98, 0]` and vips returns `[242, 98,
/// 0]`. All eight of that fixture's distinct colours land on vips's answer.
#[cfg(feature = "jp2k")]
fn sycc_to_rgb(y: u32, cb: u32, cr: u32, precision: u32) -> [u32; 3] {
    let upper = ((1i64 << precision) - 1) as f32;
    let offset = (1i64 << (precision - 1)) as f32;
    let (y, cb, cr) = (y as f32, cb as f32 - offset, cr as f32 - offset);
    let clamp = |v: f32| v.clamp(0.0, upper) as u32;
    [
        clamp(y + (1.402 * cr) as i32 as f32),
        clamp(y - (0.344 * cb + 0.714 * cr) as i32 as f32),
        clamp(y + (1.772 * cb) as i32 as f32),
    ]
}

/// The sample carrier for a band count and an element width.
///
/// The mapping is [`PixelFormat::with_channels`]'s, so 1, 3 and 4 bands reach
/// the named `Gray` / `Rgb` / `Rgba` variants and everything else reaches the
/// multiband ones. A zero band count is the one thing that has no carrier, and
/// it is unreachable from a valid codestream because `SIZ` refuses `Csiz = 0`
/// before this is called; the check is here anyway because the count comes
/// from the decoder rather than from `SIZ`, and a palette resolves to
/// whatever the palette says.
#[cfg(feature = "jp2k")]
fn carrier(bands: u32, element_bytes: u64) -> Result<PixelFormat, Jp2kError> {
    let max = u32::from(u16::MAX);
    usize::try_from(bands)
        .ok()
        .and_then(|n| PixelFormat::with_channels(n, element_bytes as usize))
        .ok_or(Jp2kError::UnsupportedBandCount { bands, max })
}

/// The interpretation tag, from the colour space the decoder resolved.
///
/// `jp2kload` reads the `colr` box's `EnumCS` and this reads the decoder's
/// answer to the same question; measured, they agree on all 22 decodable
/// fixtures and would disagree on a synthetic file whose `EnumCS` contradicts
/// its component count (issue #767).
///
/// The unresolved arms follow vips's own fallback, which guesses from the band
/// count when the colour space is unspecified: measured, a 2-band file comes
/// back `b-w` and a 5-band one comes back `srgb`, so the split is at three
/// bands and not at four.
#[cfg(feature = "jp2k")]
fn interpretation(colour: &hayro_jpeg2000::ColorSpace, element_bytes: u64) -> Interpretation {
    use hayro_jpeg2000::ColorSpace;
    let wide = element_bytes > 1;
    match colour {
        ColorSpace::Gray => {
            if wide {
                Interpretation::Grey16
            } else {
                Interpretation::Bw
            }
        }
        ColorSpace::CMYK => Interpretation::Cmyk,
        ColorSpace::RGB => {
            if wide {
                Interpretation::Rgb16
            } else {
                Interpretation::Srgb
            }
        }
        ColorSpace::Unknown { num_channels } | ColorSpace::Icc { num_channels, .. } => {
            match (*num_channels < 3, wide) {
                (true, true) => Interpretation::Grey16,
                (true, false) => Interpretation::Bw,
                (false, true) => Interpretation::Rgb16,
                (false, false) => Interpretation::Srgb,
            }
        }
    }
}

/// The interpretation an enumerated `colr` colour space maps to, or `None`
/// when openjpeg does not recognise the value.
///
/// This is the whole of #767: `jp2kload` reads the `colr` box's `EnumCS` and
/// **not** the component count, so a one-component file tagged CMYK is `cmyk`
/// and a three-component one tagged greyscale is `b-w`. Taking the decoder's
/// resolved colour space instead agrees with vips on every ordinary file and
/// disagrees on exactly those.
///
/// The recognised set is openjpeg's five, and it is measured rather than read
/// out of `opj_jp2_read_colr`: `oracle-captures/foreign-jp2k/oracle.json`'s
/// `colour_space_to_interpretation` sweeps seven values over three shapes, and
/// 14 (CIELab) behaves exactly like the undefined 99, which is what says both
/// fall through to UNSPECIFIED. Two independent unrecognised values agreeing
/// is what makes `None` a fallback rather than a special case for one number.
///
/// The element width picks between the flavours, which is the second half of
/// the measurement: the enum that gives `b-w` and `srgb` on an 8-bit file
/// gives `grey16` and `rgb16` on a 16-bit one.
#[cfg(feature = "jp2k")]
#[allow(dead_code)] // wired into the decode by the fix commit
fn enumerated_interpretation(enumcs: u32, element_bytes: u64) -> Option<Interpretation> {
    let wide = element_bytes > 1;
    Some(match enumcs {
        // 12 is CMYK, and it is the one answer the element width does not
        // change: vips reports `cmyk` for the 8-bit and the 16-bit file alike.
        12 => Interpretation::Cmyk,
        // 17 is greyscale. Three components tagged with it come back `b-w`,
        // which is the row that proves the band count is not consulted.
        17 if wide => Interpretation::Grey16,
        17 => Interpretation::Bw,
        // 16 sRGB, 18 sYCC and 24 e-YCC all land on the RGB tag. The last two
        // also turn the inverse YCC on, which is the decoder's job and not
        // this function's.
        16 | 18 | 24 if wide => Interpretation::Rgb16,
        16 | 18 | 24 => Interpretation::Srgb,
        _ => return None,
    })
}

/// Map the codec's decode failure onto [`Jp2kError`].
///
/// `hayro-jpeg2000` has a rich error enum of its own (a format layer, a
/// validation layer, a marker layer and a decoding layer) and every arm of its
/// `Display` renders the whole chain, so flattening to a message loses
/// nothing a caller could have matched on that this module does not already
/// type itself.
#[cfg(feature = "jp2k")]
fn decode_error(error: hayro_jpeg2000::DecodeError) -> Jp2kError {
    Jp2kError::Decode {
        message: error.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------------

/// The `jp2k`-feature-on body of [`Raster::encode_jp2k`].
#[cfg(feature = "jp2k")]
fn encode(raster: &Raster, options: SaveOptions) -> Result<Vec<u8>, EncodeError> {
    use openjpeg2_pure::{EncodeOptions, Encoder, Format, Image, ImageComponent};

    let SaveOptions { compression } = options;
    let (precision, element_bytes) = sample_depth(raster.format())?;
    let (width, height) = (raster.width(), raster.height());
    let bands = raster.format().channels();
    // `jp2ksave` writes any band count and `openjpeg2-pure-rs` encodes any
    // band count: measured, a five-band file written here reads back through
    // `vips jp2kload` bit for bit as `5 bands, srgb`. What cannot read it back
    // is `decode_jp2k`, because `hayro-jpeg2000` refuses a component set it
    // cannot map onto greyscale, RGB, CMYK or one of those plus alpha
    // (`ValidationError::TooManyChannels`). An encoder that writes files its
    // own loader rejects is a worse trade than a typed refusal, which is the
    // call `crate::jxl` makes for the band counts `zune-jpegxl` has no
    // spelling for. Lifting both halves is issue #769.
    if bands > MAX_BANDS {
        return Err(EncodeError::encode(format!(
            "jp2k: this build writes at most {MAX_BANDS} bands and the raster has \
             {bands}; vips jp2ksave writes more, but decode_jp2k cannot read them back, \
             so the refusal keeps the codec symmetric (issue #769)"
        )));
    }

    // One plane per band, de-interleaved, because JPEG 2000 is a
    // component-planar format and the encoder takes it that way.
    let data = raster.data();
    let plane = width as usize * height as usize;
    let mut components = Vec::with_capacity(bands);
    for band in 0..bands {
        let mut samples = Vec::with_capacity(plane);
        for i in 0..plane {
            let at = (i * bands + band) * element_bytes;
            samples.push(match element_bytes {
                1 => i32::from(data[at]),
                _ => i32::from(u16::from_ne_bytes([data[at], data[at + 1]])),
            });
        }
        components.push(
            ImageComponent::new(width, height, precision, false, samples)
                .map_err(|e| EncodeError::encode(format!("jp2k: component {band}: {e:?}")))?,
        );
    }

    let colour = encoder_colour_space(raster);
    let image = Image::new(width, height, colour, components)
        .map_err(|e| EncodeError::encode(format!("jp2k: {e:?}")))?;

    let (irreversible, rates) = match compression {
        // `tcp_rates[0] = 0` under `cp_disto_alloc` is OpenJPEG's spelling for
        // "no rate target", which with the reversible 5/3 wavelet is a
        // lossless encode.
        Compression::Lossless => (false, vec![0.0f32]),
        Compression::Lossy { ratio } => (true, vec![f32::from(ratio.get())]),
    };
    let encoded = Encoder::encode(
        &image,
        &EncodeOptions {
            // `jp2ksave` hard-codes `OPJ_CODEC_JP2` and writes the same
            // container for all five suffixes it registers, so there is
            // nothing here for a caller to choose.
            format: Format::Jp2,
            // libviprs schedules its own work in the engine, so a codec that
            // starts a second pool underneath it is not something this crate
            // wants. Same call as `exr`'s and `jxl-oxide`'s dropped `rayon`.
            threads: 0,
            irreversible,
            // `jp2ksave` sets `tcp_mct` from the band count alone
            // (`image->Bands >= 3`), CMYK included: measured, `mct` is 1 in
            // the `cmyk_lossless.jp2` and `rgba_lossless.jp2` fixtures' `COD`
            // segments. The reversible multiple-component transform is
            // exactly invertible, so this costs no accuracy: measured, every
            // carrier round-trips through `vips jp2kload` bit for bit with it
            // on.
            use_mct: bands >= 3,
            rates,
            num_resolutions: Some(num_resolutions(width, height)),
        },
    )
    .map_err(|e| EncodeError::encode(format!("jp2k: {e:?}")))?;
    Ok(encoded)
}

/// `jp2ksave`'s resolution count: `max(1, floor(log2(min(w, h))) - 5)`.
///
/// The floor is measured rather than read off the source. Over ten sizes,
/// `floor` agrees with `vips jp2ksave` at every one and `ceil` disagrees at
/// four of them (65, 100, 129 and 1000), which is what makes this a pin rather
/// than a guess.
#[cfg(feature = "jp2k")]
fn num_resolutions(width: u32, height: u32) -> i32 {
    let smallest = width.min(height).max(1);
    (smallest.ilog2() as i32 - 5).max(1)
}

/// The component precision and element width for a raster carrier.
///
/// # Errors
///
/// [`EncodeError::Encode`] for the float carriers, which is what
/// `vips jp2ksave` does with a `float` or `double` image: it fails with `not
/// an integer format` rather than quantising behind the caller's back.
#[cfg(feature = "jp2k")]
fn sample_depth(format: PixelFormat) -> Result<(u8, usize), EncodeError> {
    match format {
        PixelFormat::Gray8 | PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Multi8(_) => {
            Ok((8, 1))
        }
        PixelFormat::Gray16
        | PixelFormat::Rgb16
        | PixelFormat::Rgba16
        | PixelFormat::Multi16(_) => Ok((16, 2)),
        PixelFormat::RgbaF32 | PixelFormat::FloatF32(_) => Err(EncodeError::encode(format!(
            "jp2k: JPEG 2000 stores integer samples and {format:?} is float; cast to an \
             integer format first, so the quantisation is yours rather than the encoder's \
             (vips jp2ksave refuses the same image with `not an integer format`)"
        ))),
    }
}

/// The `colr` box's enumerated colour space, chosen the way `jp2ksave` chooses
/// it: from the raster's interpretation, not from its band count.
///
/// A four-band CMYK raster and a four-band RGBA one are the same
/// [`PixelFormat`] and want different boxes, which is why this reads
/// [`Raster::interpretation`] rather than `format().channels()`. The unlabelled
/// tail follows vips's own guess, which is `b-w` under three bands and `srgb`
/// at or above it: measured, a 2-band file it writes reads back `b-w` and a
/// 5-band one reads back `srgb`.
#[cfg(feature = "jp2k")]
fn encoder_colour_space(raster: &Raster) -> openjpeg2_pure::ColorSpace {
    use openjpeg2_pure::ColorSpace;
    match raster.interpretation() {
        Interpretation::Cmyk => ColorSpace::Cmyk,
        Interpretation::Bw | Interpretation::Grey16 => ColorSpace::Greyscale,
        _ if raster.format().channels() < 3 => ColorSpace::Greyscale,
        _ => ColorSpace::Srgb,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Named explicitly rather than taken from the glob: the parent's imports
    // are behind the feature and these two are used in both builds.
    use crate::pixel::PixelFormat;
    use crate::raster::Raster;

    /// The committed oracle fixtures, embedded rather than read.
    ///
    /// `include_bytes!` instead of `std::fs::read`, which is what
    /// `src/exr.rs` and `src/fits.rs` use, for one reason: the Miri gate
    /// aborts the whole run on the first filesystem call its isolation layer
    /// refuses, so every filesystem-touching test has to be recorded in
    /// `tests/miri_fs_test_inventory.txt` and annotated. These bytes are the
    /// same bytes either way, so taking them at compile time keeps the codec
    /// tests runnable under Miri and off that ledger.
    macro_rules! fixtures {
        ($($file:literal),* $(,)?) => {
            /// Every fixture in `oracle-captures/foreign-jp2k/fixtures/`, by
            /// name.
            const FIXTURES: &[(&str, &[u8])] = &[$((
                $file,
                include_bytes!(concat!(
                    "../oracle-captures/foreign-jp2k/fixtures/",
                    $file
                )),
            )),*];
        };
    }

    fixtures![
        "chroma_sub_off.jp2",
        "chroma_sub_on.jp2",
        "chroma_tiny_sub_on.jp2",
        "cmyk_lossless.jp2",
        "depth10u.j2k",
        "depth12s.j2k",
        "depth12u.j2k",
        "depth14u.j2k",
        "depth16u.j2k",
        "depth2u.j2k",
        "depth4u.j2k",
        "depth8u.j2k",
        "grey_tile8.jp2",
        "icc_colr.jp2",
        "int31.jp2",
        "not_jp2k.bin",
        "origin57.j2k",
        "res3.j2k",
        "rgb_lossless.jp2",
        "rgb_lossy_q48.jp2",
        "rgba_lossless.jp2",
        "sub420.j2k",
        "truncated_at_codestream.jp2",
        "truncated_in_boxes.jp2",
        "truncated_in_siz.jp2",
        "truncated_in_tile.jp2",
        "uint31.jp2",
        "zeroed_body.jp2",
    ];

    /// One committed fixture's bytes.
    fn fixture(name: &str) -> &'static [u8] {
        FIXTURES
            .iter()
            .find(|(n, _)| *n == name)
            .unwrap_or_else(|| panic!("fixture {name} must be in the embedded table"))
            .1
    }

    /// Decode a fixture with the default limits, panicking with the fixture's
    /// name rather than with a bare error.
    #[cfg(feature = "jp2k")]
    fn decoded(name: &str) -> Raster {
        decode_jp2k(fixture(name), DecodeLimits::default())
            .unwrap_or_else(|e| panic!("{name} must decode: {e}"))
    }

    /// The SHA-256 of a raster's buffer, which is the shape every
    /// `decoded_raster.sha256` in `oracle.json` is in: `vips rawsave` writes
    /// exactly the native-endian interleaved samples a `Raster` holds.
    #[cfg(feature = "jp2k")]
    fn payload_digest(raster: &Raster) -> String {
        use sha2::Digest;
        crate::hex::hex_lower(&sha2::Sha256::digest(raster.data()))
    }

    /// A raster's samples as `u32`, whatever the carrier width, in the
    /// band-interleaved order the oracle's `getpoint_all` uses.
    #[cfg(feature = "jp2k")]
    fn samples(raster: &Raster) -> Vec<u32> {
        let data = raster.data();
        match raster.format().bytes_per_channel() {
            1 => data.iter().map(|b| u32::from(*b)).collect(),
            _ => data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| u32::from(u16::from_ne_bytes(*c)))
                .collect(),
        }
    }

    /// A metadata field as an integer, or `None` when it is absent.
    #[cfg(feature = "jp2k")]
    fn int_field(raster: &Raster, key: &str) -> Option<i64> {
        match raster.get_field(key) {
            Some(MetadataValue::Int(v)) => Some(v),
            _ => None,
        }
    }

    // -----------------------------------------------------------------------
    // The decode, against the vips oracle
    // -----------------------------------------------------------------------

    /// One committed fixture and everything `vipsheader -a` and
    /// `vips rawsave` said about it on vips 8.18.6.
    ///
    /// Every number is transcribed from
    /// `oracle-captures/foreign-jp2k/oracle.json`, which `capture.py` wrote by
    /// running the binary; none of it was derived from reading `jp2kload.c`.
    #[cfg(feature = "jp2k")]
    struct Pin {
        fixture: &'static str,
        width: u32,
        height: u32,
        bands: usize,
        wide: bool,
        interpretation: Interpretation,
        /// The SHA-256 of `vips rawsave`'s output, which is
        /// `decoded_raster.sha256` in the capture.
        payload: &'static str,
        /// `bits-per-sample`.
        bits: i64,
        /// `n-pages` in the capture, which travels here as
        /// `jp2k-resolutions`.
        resolutions: i64,
    }

    /// The fixtures whose decode is byte-identical to vips's.
    ///
    /// Every one of these is a reversible 5/3 codestream, and the reversible
    /// wavelet is integer-specified, so there is no tolerance here at all: a
    /// digest either matches or the port is wrong. The four irreversible
    /// fixtures are in `the_irreversible_fixtures_agree_with_vips_to_within_four_counts`
    /// instead, and `origin57.j2k` is in
    /// `the_image_origin_is_the_one_divergence_on_geometry`.
    #[cfg(feature = "jp2k")]
    const EXACT: &[Pin] = &[
        Pin {
            fixture: "rgb_lossless.jp2",
            width: 4,
            height: 3,
            bands: 3,
            wide: false,
            interpretation: Interpretation::Srgb,
            payload: "c875696bdedeac8d1e24565c11a2eac3ba29bb7727596c1ed159983d9a21c21d",
            bits: 8,
            resolutions: 1,
        },
        Pin {
            fixture: "rgba_lossless.jp2",
            width: 4,
            height: 3,
            bands: 4,
            wide: false,
            interpretation: Interpretation::Srgb,
            payload: "542d0234c2a78b9c4cb603ccce5d1cf84866f4dc3730b7988583e7ed0b09b1e7",
            bits: 8,
            resolutions: 1,
        },
        Pin {
            fixture: "cmyk_lossless.jp2",
            width: 4,
            height: 3,
            bands: 4,
            wide: false,
            interpretation: Interpretation::Cmyk,
            payload: "542d0234c2a78b9c4cb603ccce5d1cf84866f4dc3730b7988583e7ed0b09b1e7",
            bits: 8,
            resolutions: 1,
        },
        Pin {
            fixture: "grey_tile8.jp2",
            width: 37,
            height: 21,
            bands: 1,
            wide: false,
            interpretation: Interpretation::Bw,
            payload: "20c9ed48efcf557610597f53190568ce910d3e67ea8970a38b379fa7d6e97343",
            bits: 8,
            resolutions: 1,
        },
        Pin {
            fixture: "icc_colr.jp2",
            width: 4,
            height: 3,
            bands: 3,
            wide: false,
            interpretation: Interpretation::Srgb,
            payload: "c875696bdedeac8d1e24565c11a2eac3ba29bb7727596c1ed159983d9a21c21d",
            bits: 8,
            resolutions: 1,
        },
        Pin {
            fixture: "res3.j2k",
            width: 32,
            height: 24,
            bands: 1,
            wide: false,
            interpretation: Interpretation::Bw,
            payload: "2f7c22b8a227f529e1a1e65bc101471bff7f1e97a5098a3700532c12f486b375",
            bits: 8,
            resolutions: 3,
        },
        Pin {
            fixture: "sub420.j2k",
            width: 8,
            height: 4,
            bands: 3,
            wide: false,
            interpretation: Interpretation::Srgb,
            payload: "5a59a2bd1b802c306be83997392db407de7cc46d8c1ec2da11fd98d48c4ce0b1",
            bits: 8,
            resolutions: 1,
        },
    ];

    /**
     * Pins the reversible half of the format against the vips oracle as a
     * whole payload rather than as a spot check: for seven committed
     * fixtures the decoded buffer is byte-identical to what `vips rawsave`
     * wrote, digest for digest, with no tolerance anywhere, and the
     * geometry, band count, carrier width, interpretation, `bits-per-sample`
     * and resolution count all agree too.
     * The reversible 5/3 wavelet is integer-specified, so this is a parity
     * claim and not an approximation: a difference here is a bug rather
     * than rounding. Three of the seven would pass on geometry alone with
     * a completely wrong picture (`sub420.j2k` needs the inverse YCC,
     * `grey_tile8.jp2` needs the tile seams to line up, `cmyk_lossless.jp2`
     * needs four bands in the right order), which is why the assertion is
     * on the payload.
     * Input: seven fixtures from `oracle-captures/foreign-jp2k/fixtures/`
     * -> Output: the `decoded_raster.sha256` captured for each.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn the_reversible_fixtures_decode_to_the_bytes_vips_produces() {
        for pin in EXACT {
            let raster = decoded(pin.fixture);
            assert_eq!(
                (raster.width(), raster.height()),
                (pin.width, pin.height),
                "{}: geometry",
                pin.fixture
            );
            assert_eq!(
                raster.format().channels(),
                pin.bands,
                "{}: band count",
                pin.fixture
            );
            assert_eq!(
                raster.format().bytes_per_channel(),
                if pin.wide { 2 } else { 1 },
                "{}: carrier width",
                pin.fixture
            );
            assert_eq!(
                raster.interpretation(),
                pin.interpretation,
                "{}: interpretation",
                pin.fixture
            );
            assert_eq!(
                payload_digest(&raster),
                pin.payload,
                "{}: the decoded buffer must be the bytes vips rawsave wrote, and the \
                 reversible wavelet leaves no room for it to be nearly right",
                pin.fixture
            );
            assert_eq!(
                int_field(&raster, "bits-per-sample"),
                Some(pin.bits),
                "{}: bits-per-sample",
                pin.fixture
            );
            assert_eq!(
                int_field(&raster, "jp2k-resolutions"),
                Some(pin.resolutions),
                "{}: the resolution count vips reports as n-pages",
                pin.fixture
            );
        }
    }

    /**
     * Pins the irreversible half against vips with the tolerance the
     * measurement produced and not one count more: the 9/7 wavelet is
     * float-specified, so `hayro-jpeg2000` and OpenJPEG are entitled to
     * disagree in the last place, and each fixture carries its own measured
     * worst deviation rather than a shared slack number. A change that makes
     * any one of them worse goes red even though the others still fit.
     * Each case also asserts its decode is NOT byte-identical to the captured
     * vips payload, which is the positive control: without it a port that
     * became exact would leave a tolerance nobody re-measured, and without
     * the tolerance being per fixture the 1-count `chroma_tiny_sub_on` could
     * drift to 4 unnoticed.
     * The numbers are the worst deviation over the pixels the capture
     * recorded, which for the two 16x16 fixtures is six pixels rather than
     * the whole image. Re-measured live against `vips rawsave` over every
     * pixel, the whole-image maxima are 4, 1, 2 and 3; `chroma_sub_on`'s
     * third count sits outside the six pinned pixels, which is why its number
     * here is 2 and not 3.
     * Input: the four irreversible fixtures -> Output: at every pixel the
     * capture recorded, a per-band deviation no larger than the measured one.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn the_irreversible_fixtures_agree_with_vips_to_within_the_measured_tolerance() {
        // Every `want` below is `getpoint` or `getpoint_all` out of
        // `oracle-captures/foreign-jp2k/oracle.json`, which `capture.py` read
        // off `vips getpoint` on 8.18.6. The first two record every pixel in
        // the image; the two 16x16 ones record the six the capture pinned.
        struct Lossy {
            fixture: &'static str,
            width: u32,
            /// The captured `decoded_raster.sha256`, used as a negative
            /// control: an exact decode would match it.
            payload: &'static str,
            /// `((x, y), the bands vips reported there)`.
            points: &'static [((u32, u32), &'static [u32])],
            /// The worst per-band disagreement measured on this branch.
            tolerance: u32,
        }
        let cases: &[Lossy] = &[
            Lossy {
                fixture: "rgb_lossy_q48.jp2",
                width: 4,
                payload: "f1ce4d4f51a197a6d886a89dc0d08432615b076e47bd2dcaee369018645ecab5",
                points: &[
                    ((0, 0), &[0, 0, 0]),
                    ((1, 0), &[59, 98, 31]),
                    ((2, 0), &[120, 194, 59]),
                    ((3, 0), &[184, 36, 86]),
                    ((0, 1), &[13, 151, 210]),
                    ((1, 1), &[75, 248, 240]),
                    ((2, 1), &[137, 89, 15]),
                    ((3, 1), &[198, 185, 43]),
                    ((0, 2), &[28, 45, 165]),
                    ((1, 2), &[85, 145, 195]),
                    ((2, 2), &[146, 241, 223]),
                    ((3, 2), &[209, 80, 254]),
                ],
                tolerance: 4,
            },
            Lossy {
                fixture: "chroma_tiny_sub_on.jp2",
                width: 4,
                payload: "6cb658b9f2891816ced56e1e8222b8505d9121f5651e44d71541b14e2d2ddb75",
                points: &[
                    ((0, 0), &[75, 75, 75]),
                    ((1, 0), &[149, 149, 149]),
                    ((2, 0), &[29, 29, 29]),
                    ((3, 0), &[225, 225, 225]),
                    ((0, 1), &[179, 179, 179]),
                    ((1, 1), &[105, 105, 105]),
                    ((2, 1), &[0, 0, 0]),
                    ((3, 1), &[255, 255, 255]),
                ],
                tolerance: 1,
            },
            Lossy {
                fixture: "chroma_sub_off.jp2",
                width: 16,
                payload: "43bb34516256347c9d1be4189eed2c49ed8f5cb8ee9548a09fe9470785935797",
                points: &[
                    ((0, 0), &[0, 1, 255]),
                    ((1, 0), &[17, 0, 239]),
                    ((2, 0), &[36, 0, 221]),
                    ((3, 0), &[51, 0, 204]),
                    ((0, 1), &[2, 17, 255]),
                    ((15, 15), &[255, 254, 0]),
                ],
                tolerance: 2,
            },
            Lossy {
                fixture: "chroma_sub_on.jp2",
                width: 16,
                payload: "e75b9c59444f2dca463ce0902e9c5cac50120d48c1052baaf1b02e5c124a4657",
                points: &[
                    ((0, 0), &[4, 1, 241]),
                    ((1, 0), &[6, 3, 243]),
                    ((2, 0), &[35, 2, 205]),
                    ((3, 0), &[37, 4, 207]),
                    ((0, 1), &[14, 11, 251]),
                    ((15, 15), &[253, 251, 16]),
                ],
                tolerance: 2,
            },
        ];

        for case in cases {
            let raster = decoded(case.fixture);
            assert_ne!(
                payload_digest(&raster),
                case.payload,
                "{}: this fixture is in the irreversible table because its decode is NOT \
                 byte-identical to vips. If it has become identical, move it into EXACT \
                 rather than leaving a tolerance nothing needs",
                case.fixture
            );
            let got = samples(&raster);
            let bands = raster.format().channels();
            let mut worst = 0u32;
            for ((x, y), want) in case.points {
                let at = ((*y * case.width + *x) as usize) * bands;
                let mine = &got[at..at + bands];
                assert_eq!(
                    mine.len(),
                    want.len(),
                    "{}: band count at ({x}, {y})",
                    case.fixture
                );
                for (band, (mine, want)) in mine.iter().zip(want.iter()).enumerate() {
                    let delta = mine.abs_diff(*want);
                    assert!(
                        delta <= case.tolerance,
                        "{}: band {band} at ({x}, {y}) is {mine} where vips says {want}, \
                         {delta} counts out against a measured tolerance of {}",
                        case.fixture,
                        case.tolerance
                    );
                    worst = worst.max(delta);
                }
            }
            assert_eq!(
                worst, case.tolerance,
                "{}: the tolerance is a measurement, not slack. The worst disagreement \
                 over the pinned points is now {worst} where {} was measured; re-measure \
                 and move the number rather than leaving it wide",
                case.fixture, case.tolerance
            );
        }
    }

    /**
     * Pins the left-justification, which is the one loader behaviour an
     * 8-bit-only test cannot catch: `jp2kload` shifts a precision-N sample
     * left by `element_bits - N`, so depths 8 and 16 shift by zero and are
     * the identity while every other depth is not.
     * Seven depths from 2 to 16 bits, each with the five samples the
     * capture put in and the five vips gave back. Depths 8 and 16 are in the
     * table on purpose, as the positive control for the other five: they
     * prove the pipeline reads the right samples in the first place, so a
     * failure at depth 10 is the shift and not the decode.
     * Input: the seven unsigned depth fixtures -> Output: the `out` array
     * `oracle-captures/foreign-jp2k/oracle.json` recorded for each, plus the
     * carrier vips chose and the true depth in `bits-per-sample`.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn a_component_narrower_than_its_element_is_left_justified() {
        // (fixture, declared precision, the carrier vips chose, the samples
        // vips read back). Every row is from the `bit_depth_is_left_justified`
        // record; the `in` column it also carries is what opj_compress wrote.
        let cases: [(&str, i64, PixelFormat, [u32; 5]); 7] = [
            ("depth2u.j2k", 2, PixelFormat::Gray8, [0, 64, 64, 128, 192]),
            ("depth4u.j2k", 4, PixelFormat::Gray8, [0, 16, 64, 224, 240]),
            ("depth8u.j2k", 8, PixelFormat::Gray8, [0, 1, 64, 254, 255]),
            (
                "depth10u.j2k",
                10,
                PixelFormat::Gray16,
                [0, 64, 16384, 65408, 65472],
            ),
            (
                "depth12u.j2k",
                12,
                PixelFormat::Gray16,
                [0, 16, 16384, 65504, 65520],
            ),
            (
                "depth14u.j2k",
                14,
                PixelFormat::Gray16,
                [0, 4, 16384, 65528, 65532],
            ),
            (
                "depth16u.j2k",
                16,
                PixelFormat::Gray16,
                [0, 1, 16384, 65534, 65535],
            ),
        ];
        for (name, precision, format, want) in cases {
            let raster = decoded(name);
            assert_eq!(raster.format(), format, "{name}: carrier");
            assert_eq!(
                samples(&raster),
                want.to_vec(),
                "{name}: a {precision}-bit sample must be shifted left into its element, \
                 which is what puts the real depth in bits-per-sample and not in the value"
            );
            assert_eq!(
                int_field(&raster, "bits-per-sample"),
                Some(precision),
                "{name}: the true depth survives only here"
            );
        }
    }

    /**
     * Pins the two carrier refusals, both of which are gaps in this crate
     * rather than in the format, and both of which would otherwise be
     * silently wrong answers.
     * A signed component has no `PixelFormat` to go into and
     * `hayro-jpeg2000` reports every component DC-level-shifted into the
     * unsigned range, so decoding one anyway comes back offset by half the
     * range: measured on `depth12s.j2k`, 0 where vips says -32768. A 31-bit
     * component has no 32-bit integer carrier and does not survive the
     * decoder's `f32` container either.
     * Each refusal has a positive control beside it, and they are the point:
     * `depth12u.j2k` is the same file with the sign bit clear and
     * `depth16u.j2k` is the same shape one bit under the ceiling, so a
     * refusal that fired for the wrong reason would take its control down
     * with it.
     * Input: `depth12s.j2k`, `int31.jp2`, `uint31.jp2` -> Output: the typed
     * variant naming the component and the number, and a clean decode for
     * each control.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn a_carrier_this_crate_does_not_have_is_a_typed_refusal_and_not_an_offset() {
        let err = decode_jp2k(fixture("depth12s.j2k"), DecodeLimits::default())
            .expect_err("a signed component has no carrier");
        assert!(
            matches!(
                err,
                SourceError::Jp2k(Jp2kError::SignedComponent {
                    component: 0,
                    precision: 12
                })
            ),
            "a signed component must be refused by name: {err:?}"
        );
        // The control: the same 5x1 shape at the same precision with the sign
        // bit clear decodes, so the refusal above is the sign bit and not the
        // depth or the geometry.
        assert_eq!(decoded("depth12u.j2k").width(), 5);

        // `uint31.jp2` is the precision refusal on its own. Its sibling
        // `int31.jp2` is signed AND 31-bit and is refused by the sign bit
        // first, which is the order the checks run in and is asserted here so
        // the two cannot be confused for one another.
        let err = decode_jp2k(fixture("uint31.jp2"), DecodeLimits::default())
            .expect_err("31 bits per sample has no carrier");
        let SourceError::Jp2k(Jp2kError::PrecisionNotSupported {
            component,
            precision,
            max,
        }) = err
        else {
            panic!("uint31.jp2 must be refused by precision: {err:?}");
        };
        assert_eq!((component, precision, max), (0, 31, MAX_PRECISION));

        let err = decode_jp2k(fixture("int31.jp2"), DecodeLimits::default())
            .expect_err("a signed 31-bit component has neither carrier");
        assert!(
            matches!(
                err,
                SourceError::Jp2k(Jp2kError::SignedComponent { precision: 31, .. })
            ),
            "int31.jp2 is signed and 31-bit; the sign bit is checked first: {err:?}"
        );
        // The control: one bit under the ceiling still decodes.
        assert_eq!(decoded("depth16u.j2k").format(), PixelFormat::Gray16);
    }

    /**
     * Pins the inverse YCC, which is the one place a decode that looks
     * plausible is completely wrong: `hayro-jpeg2000` hands the raw Y, Cb
     * and Cr planes back for a bare codestream with subsampled chroma, and
     * `jp2kload` runs OpenJPEG's `sycc_to_rgb` over them first. Without the
     * transform `sub420.j2k`'s first pixel is `[128, 16, 240]` where vips
     * says `[255, 87, 0]`.
     * Asserted through the module's own `sycc_to_rgb` as well as through the
     * decode, because the arithmetic is the claim: OpenJPEG's coefficients
     * with truncating casts give `[242, 98, 0]` at the third pixel where a
     * rounding implementation gives `[243, 98, 0]` and vips gives 242. That
     * single count is the whole difference between the transform being right
     * and being nearly right, and the whole-payload digest in
     * `the_reversible_fixtures_decode_to_the_bytes_vips_produces` is what
     * catches it.
     * Input: the eight distinct colours of `sub420.j2k` -> Output: the
     * `getpoint_all` the capture recorded.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn a_bare_codestream_with_subsampled_chroma_gets_openjpegs_inverse_ycc() {
        // The first row of the capture's `getpoint_all`, which repeats in
        // pairs because the chroma is halved on both axes and upsampled by
        // replication.
        let want: [[u32; 3]; 8] = [
            [255, 87, 0],
            [255, 87, 0],
            [242, 98, 0],
            [242, 98, 0],
            [200, 109, 36],
            [200, 109, 36],
            [158, 120, 90],
            [158, 120, 90],
        ];
        let raster = decoded("sub420.j2k");
        let got = samples(&raster);
        for (x, want) in want.iter().enumerate() {
            assert_eq!(
                &got[x * 3..x * 3 + 3],
                want.as_slice(),
                "sub420.j2k pixel ({x}, 0)"
            );
        }

        // And the arithmetic on its own, at the source planes the capture
        // recorded: Y flat at 128, Cb 16 and 46, Cr 240 and 210.
        assert_eq!(sycc_to_rgb(128, 16, 240, 8), [255, 87, 0]);
        assert_eq!(
            sycc_to_rgb(128, 46, 210, 8),
            [242, 98, 0],
            "the casts truncate: rounding gives 243 here and vips gives 242"
        );
    }

    /**
     * Pins the ICC profile, which arrives through this module's own box walk
     * rather than through the decoder: `jp2kload` copies a METH=2 `colr`
     * payload verbatim without validating it, and the capture's fixture
     * carries 24 bytes that are not an ICC profile at all, so
     * `hayro-jpeg2000` drops them and vips keeps them.
     * The control is the same image with a METH=1 box, which must carry no
     * profile: without it a walk that attached the four EnumCS bytes to
     * every file would pass.
     * Input: `icc_colr.jp2` and `rgb_lossless.jp2` -> Output: exactly the 24
     * injected bytes, and no field at all.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn a_meth_2_colr_box_is_copied_out_verbatim_and_unvalidated() {
        let raster = decoded("icc_colr.jp2");
        let profile = raster
            .get_field("icc-profile-data")
            .expect("the METH=2 payload must reach icc-profile-data");
        let MetadataValue::Blob(bytes) = profile else {
            panic!("a profile is a blob");
        };
        // `injected_profile` in the capture, which is 16..=39 and is not a
        // valid ICC profile, deliberately.
        assert_eq!(bytes, (16u8..40).collect::<Vec<u8>>());

        assert!(
            decoded("rgb_lossless.jp2")
                .get_field("icc-profile-data")
                .is_none(),
            "a METH=1 colr box is an enumerated colour space and not a profile"
        );
    }

    /**
     * Pins the two metadata keys whose presence is conditional, both of
     * which a port that always attaches them gets wrong on every small
     * image.
     * `tile-width` and `tile-height` appear only when the image is more than
     * one tile: `grey_tile8.jp2` is 37x21 on an 8x8 grid and carries them,
     * and `rgb_lossless.jp2` at the default 512 is one tile and carries
     * neither. And the resolution count travels as `jp2k-resolutions`, never
     * as `n-pages`, because vips's `page` selects a resolution level rather
     * than a frame and this crate reserves that key (issue #635).
     * Input: `grey_tile8.jp2` and `rgb_lossless.jp2` -> Output: 8 and 8, and
     * nothing.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn the_tile_geometry_is_attached_only_when_there_is_more_than_one_tile() {
        let tiled = decoded("grey_tile8.jp2");
        assert_eq!(int_field(&tiled, "tile-width"), Some(8));
        assert_eq!(int_field(&tiled, "tile-height"), Some(8));

        let single = decoded("rgb_lossless.jp2");
        assert_eq!(
            (
                single.get_field("tile-width"),
                single.get_field("tile-height")
            ),
            (None, None),
            "a one-tile image carries neither field, which is what vipsheader shows"
        );

        // The resolution count, and the key it is not under.
        let three = decoded("res3.j2k");
        assert_eq!(int_field(&three, "jp2k-resolutions"), Some(3));
        assert!(
            three.get_field("n-pages").is_none(),
            "a resolution level is not a page: `page` selects a size here, not a frame"
        );
        assert_eq!(three.get_n_pages(), 1);
    }

    /**
     * Pins that every malformed fixture is refused, that the two layers that
     * can refuse are distinguishable by variant rather than by message, and
     * that a file which is not JPEG 2000 never reaches this loader at all.
     * The layer split is the claim worth pinning: this module walks the box
     * structure itself, so a cut inside the boxes is a `Container` refusal
     * before any decoder is asked, while a cut inside the codestream is the
     * decoder's `Decode`. A port that let the box walk panic on the first of
     * those would never reach the second.
     * `truncated_in_tile.jp2` is the divergence worth naming: vips reads its
     * header fine and fails only when the pixels are pulled, where this
     * loader refuses at header time. Both refuse, so the answer a caller sees
     * through `decode_jp2k` is the same.
     * The control is `rgb_lossless.jp2`, an untruncated file of the same
     * shape, which must decode: without it "everything is refused" would pass
     * for a loader that refuses everything.
     * Input: five broken fixtures and one file that is not JPEG 2000 ->
     * Output: a typed refusal for each, and no route for the last.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn every_malformed_fixture_is_refused_and_says_which_layer_refused_it() {
        // The two the box walk answers, before any decoder is asked.
        for name in ["truncated_in_boxes.jp2", "zeroed_body.jp2"] {
            let err = decode_jp2k(fixture(name), DecodeLimits::default())
                .expect_err("a broken box structure must be refused");
            assert!(
                matches!(err, SourceError::Jp2k(Jp2kError::Container { .. })),
                "{name}: the container walk refuses this one: {err:?}"
            );
        }

        // The three the codestream answers.
        for name in [
            "truncated_at_codestream.jp2",
            "truncated_in_siz.jp2",
            "truncated_in_tile.jp2",
        ] {
            let err = decode_jp2k(fixture(name), DecodeLimits::default())
                .expect_err("a broken codestream must be refused");
            assert!(
                matches!(
                    err,
                    SourceError::Jp2k(Jp2kError::Container { .. } | Jp2kError::Decode { .. })
                ),
                "{name}: {err:?}"
            );
        }

        // The control.
        assert_eq!(decoded("rgb_lossless.jp2").width(), 4);
    }

    /**
     * The image origin is the one geometry this loader and vips disagree
     * about, and it is pinned so the disagreement is a decision on the
     * record rather than a surprise (issue #766).
     * `origin57.j2k` declares `Xsiz = 37, XOsiz = 5, Ysiz = 31, YOsiz = 7`.
     * The standard's image is `Xsiz - XOsiz` by `Ysiz - YOsiz`, which is
     * 32x24 and is what this loader and `hayro-jpeg2000` both report. vips
     * reports 27x17 with `xoffset = -5`, which is that size less the origin a
     * second time.
     * Input: `origin57.j2k` -> Output: 32x24, and explicitly not vips's
     * 27x17.
     */

    /// Rewrite a JP2's `METH = 1` `colr` box to name `enumcs`, leaving every
    /// other byte alone.
    ///
    /// This is what `capture.py`'s `retag_colr` does to build the
    /// `colour_space_to_interpretation` sweep, reproduced here so the sweep
    /// can be asserted against committed fixtures rather than against the
    /// capture's uncommitted `outputs/`.
    #[cfg(feature = "jp2k")]
    fn retagged(bytes: &[u8], enumcs: u32) -> Vec<u8> {
        let mut out = bytes.to_vec();
        let at = out
            .windows(4)
            .position(|w| w == b"colr")
            .expect("a colr box");
        assert_eq!(out[at + 4], 1, "METH must be 1 for a retag");
        out[at + 7..at + 11].copy_from_slice(&enumcs.to_be_bytes());
        out
    }

    /// Wrap a bare codestream in a minimal JP2 (signature, `ftyp`, `jp2h`
    /// holding `ihdr` and a `METH = 1` `colr`, then `jp2c`) naming `enumcs`.
    ///
    /// The one-component half of the sweep needs this: the capture built its
    /// one-component base with `vips jp2ksave` into `outputs/`, which is not
    /// committed, and the only committed one-component files are bare
    /// codestreams with no `colr` box at all.
    #[cfg(feature = "jp2k")]
    fn wrapped(codestream: &[u8], w: u32, h: u32, nc: u16, bpc: u8, enumcs: u32) -> Vec<u8> {
        fn boxed(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
            let mut b = ((payload.len() + 8) as u32).to_be_bytes().to_vec();
            b.extend_from_slice(kind);
            b.extend_from_slice(payload);
            b
        }
        let mut out: Vec<u8> = vec![0, 0, 0, 12, b'j', b'P', b' ', b' ', 0x0d, 0x0a, 0x87, 0x0a];
        out.extend(boxed(b"ftyp", b"jp2 \x00\x00\x00\x00jp2 "));
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&h.to_be_bytes());
        ihdr.extend_from_slice(&w.to_be_bytes());
        ihdr.extend_from_slice(&nc.to_be_bytes());
        ihdr.extend_from_slice(&[bpc - 1, 7, 0, 0]);
        let mut colr = vec![1u8, 0, 0];
        colr.extend_from_slice(&enumcs.to_be_bytes());
        let mut jp2h = boxed(b"ihdr", &ihdr);
        jp2h.extend(boxed(b"colr", &colr));
        out.extend(boxed(b"jp2h", &jp2h));
        out.extend(boxed(b"jp2c", codestream));
        out
    }

    /**
     * The `colr` box's enumerated colour space decides the interpretation,
     * not the component count (issue #767).
     * Every cell is `oracle-captures/foreign-jp2k/oracle.json`'s
     * `colour_space_to_interpretation` record, which `capture.py` produced by
     * retagging two codestreams with every enum openjpeg recognises and
     * reading `vipsheader` back. The two rows that make the rule a rule are
     * the ones where the enum and the band count disagree: a **one**-component
     * file tagged CMYK is `cmyk`, and a **three**-component file tagged
     * greyscale is `b-w`. A port that reads the band count gets both wrong,
     * and this loader did.
     * The element width picks between the flavours of each, which is why the
     * sweep runs 8-bit and 16-bit: the same enum that gives `b-w` and `srgb`
     * on an 8-bit file gives `grey16` and `rgb16` on a 16-bit one.
     * Input: `rgb_lossless.jp2` retagged, `depth8u.j2k` wrapped in a minimal
     * JP2, and a 16-bit RGB file this crate encodes, each at every enum ->
     * Output: the interpretation `vipsheader` reported for that cell.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn the_colr_box_enum_decides_the_interpretation_not_the_component_count() {
        // (enum, what vipsheader printed for 3 components 8-bit, for 1
        // component 8-bit, and for 3 components 16-bit). `None` is a cell
        // `hayro-jpeg2000` refuses before the tag is reached; each one is an
        // issue of its own and they are asserted separately below.
        let cells: &[(
            u32,
            Option<Interpretation>,
            Option<Interpretation>,
            Option<Interpretation>,
        )] = &[
            (
                12,
                Some(Interpretation::Cmyk),
                Some(Interpretation::Cmyk),
                Some(Interpretation::Cmyk),
            ),
            (
                14,
                Some(Interpretation::Srgb),
                None,
                Some(Interpretation::Rgb16),
            ),
            (
                16,
                Some(Interpretation::Srgb),
                Some(Interpretation::Srgb),
                Some(Interpretation::Rgb16),
            ),
            (
                17,
                Some(Interpretation::Bw),
                Some(Interpretation::Bw),
                Some(Interpretation::Grey16),
            ),
            (
                18,
                Some(Interpretation::Srgb),
                None,
                Some(Interpretation::Rgb16),
            ),
            (24, None, None, None),
            (99, None, None, None),
        ];

        let wide = Raster::new(
            4,
            3,
            PixelFormat::Rgb16,
            (0..4u32 * 3 * 3 * 2).map(|i| (i % 251) as u8).collect(),
        )
        .unwrap()
        .encode_jp2k(SaveOptions::default())
        .expect("a 16-bit RGB base for the wide half of the sweep");

        let mut checked = 0;
        for (enumcs, three_8, one_8, three_16) in cells {
            let cases: [(&str, Vec<u8>, Option<Interpretation>); 3] = [
                (
                    "3 components, 8-bit",
                    retagged(fixture("rgb_lossless.jp2"), *enumcs),
                    *three_8,
                ),
                (
                    "1 component, 8-bit",
                    wrapped(fixture("depth8u.j2k"), 5, 1, 1, 8, *enumcs),
                    *one_8,
                ),
                ("3 components, 16-bit", retagged(&wide, *enumcs), *three_16),
            ];
            for (shape, bytes, want) in cases {
                match (decode_jp2k(&bytes, DecodeLimits::default()), want) {
                    (Ok(raster), Some(want)) => {
                        assert_eq!(
                            raster.interpretation(),
                            want,
                            "EnumCS {enumcs} on {shape}: vips reports {want:?}"
                        );
                        checked += 1;
                    }
                    (Ok(raster), None) => panic!(
                        "EnumCS {enumcs} on {shape} now decodes as {:?}; the decoder used to \
                         refuse it, so #848 or #849 is fixed and this table is stale",
                        raster.interpretation()
                    ),
                    (Err(_), None) => {}
                    (Err(e), Some(want)) => {
                        panic!("EnumCS {enumcs} on {shape} must decode as {want:?}, got {e}")
                    }
                }
            }
        }
        assert_eq!(checked, 13, "the sweep has to reach every readable cell");
    }

    /**
     * An enum openjpeg does not recognise falls back to the band count, which
     * is the arm that keeps every ordinary file working (issue #767).
     * `EnumCS 14` is CIELab and openjpeg maps it to nothing, so vips guesses
     * from the component count exactly as it does for the undefined `99`.
     * Two independent unrecognised values behaving the same way is what makes
     * this a fallback rather than a fourteen-shaped special case.
     * Input: `EnumCS 14` over the three shapes -> Output: `srgb` on three
     * 8-bit components and `rgb16` on three 16-bit ones, which is the
     * band-count guess and not a CIELab tag.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn an_enum_openjpeg_does_not_recognise_falls_back_to_the_band_count() {
        let three = decode_jp2k(
            &retagged(fixture("rgb_lossless.jp2"), 14),
            DecodeLimits::default(),
        )
        .expect("CIELab on three components decodes");
        assert_eq!(
            three.interpretation(),
            Interpretation::Srgb,
            "the band count decides here, and there is no Lab tag in the answer"
        );
        assert_ne!(three.interpretation(), Interpretation::Lab);

        // The control that says the fallback is the fallback and not the enum
        // being honoured by accident: change nothing but the enum to one that
        // *is* recognised and names a different space, and the answer moves.
        let grey = decode_jp2k(
            &retagged(fixture("rgb_lossless.jp2"), 17),
            DecodeLimits::default(),
        )
        .expect("greyscale on three components decodes");
        assert_eq!(grey.interpretation(), Interpretation::Bw);
    }

    /**
     * A one-component file tagged sRGB keeps its one band and takes vips's
     * tag, which is the half of vips's answer that is not broken (issue #767).
     * This is the combination the capture calls out: openjpeg expands the
     * header to 3 bands while the tile decode still yields 1, so `vipsheader`
     * reports `3 bands, srgb` and **any pixel read fails** with `decoded
     * image does not match container`. Reproducing that would mean writing a
     * header promising pixels that never arrive.
     * The tag and the band count are independent in this crate, which its own
     * `Interpretation` doc says outright ("the pipeline does not validate that
     * the band count matches the tag, exactly as in libvips"), so honouring
     * the enum costs nothing and keeps #767's rule whole: the enum decides,
     * and the band count is not allowed back into the decision for the one
     * shape vips gets wrong.
     * Input: `depth8u.j2k` wrapped with `EnumCS 16` -> Output: one band, five
     * readable samples, tagged `srgb`.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn a_one_component_file_tagged_srgb_keeps_its_band_and_takes_the_tag() {
        let raster = decode_jp2k(
            &wrapped(fixture("depth8u.j2k"), 5, 1, 1, 8, 16),
            DecodeLimits::default(),
        )
        .expect("the file has one decodable component whatever its colr box says");

        assert_eq!(raster.format().channels(), 1, "one component, one band");
        assert_eq!(raster.interpretation(), Interpretation::Srgb);
        // The half that matters: the pixels are there. vips's 3-band answer
        // cannot be read at all.
        assert_eq!(raster.data().len(), 5);
        assert_eq!(
            raster.data(),
            decoded("depth8u.j2k").data(),
            "wrapping the same codestream in a JP2 must not change its samples"
        );
    }

    #[test]
    #[cfg(feature = "jp2k")]
    fn the_image_origin_is_the_one_divergence_on_geometry() {
        let raster = decoded("origin57.j2k");
        assert_eq!(
            (raster.width(), raster.height()),
            (32, 24),
            "Xsiz - XOsiz by Ysiz - YOsiz, which is what the standard calls the image"
        );
        assert_ne!(
            (raster.width(), raster.height()),
            (27, 17),
            "if this has become vips's answer the divergence in the module docs is stale \
             and issue #766 is fixed"
        );
    }

    /**
     * What vips's 27x17 actually is, measured rather than argued: the
     * **top-left crop** of the 32x24 this loader reports, so `jp2kload`
     * silently drops 309 of the image's 768 samples on a codestream whose
     * origin is not the grid origin (issue #766).
     * This is what decided #766. "Two libraries report different sizes" says
     * nothing about which is right; "one of them returns 40% of the picture"
     * does. Cropping our decode to vips's dimensions at (0, 0) and hashing it
     * has to reproduce the capture's `decoded_raster.sha256` byte for byte,
     * and the uncropped digest has to differ, or the crop is not what happened.
     * Input: `origin57.j2k` -> Output: our 32x24 cropped to 27x17 digests to
     * vips's payload; our full 32x24 does not.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn vips_returns_the_top_left_crop_of_what_this_loader_decodes() {
        use sha2::Digest;

        let raster = decoded("origin57.j2k");
        let (w, h) = (raster.width() as usize, raster.height() as usize);
        let full = raster.data();

        // `decoded_raster.sha256` for the `image_origin_offset` record, which
        // is `vips rawsave` over the 459 bytes `jp2kload` handed back.
        const VIPS_PAYLOAD: &str =
            "d4c1e2f8b0f763add13ededfa4881780428c177187c6302a0616719e34b81f76";

        let cropped: Vec<u8> = (0..17)
            .flat_map(|y| (0..27).map(move |x| (y, x)))
            .map(|(y, x)| full[y * w + x])
            .collect();
        assert_eq!(cropped.len(), 27 * 17);
        assert_eq!(
            crate::hex::hex_lower(&sha2::Sha256::digest(&cropped)),
            VIPS_PAYLOAD,
            "vips's whole answer is the top-left 27x17 of ours, so the 5 rightmost \
             columns and 7 bottom rows are pixels it decoded and threw away"
        );

        // The control, so "the crop matches" cannot be "everything matches".
        assert_eq!((w, h), (32, 24));
        assert_ne!(
            payload_digest(&raster),
            VIPS_PAYLOAD,
            "the uncropped decode must not digest to vips's, or there is no divergence"
        );
        assert_eq!(full.len() - cropped.len(), 309, "what vips drops");
    }

    /**
     * The image origin travels as the negative `xoffset` / `yoffset` vips
     * records for it, which is also this crate's own convention for a crop
     * (issue #766, #721).
     * `jp2kload` reports `xoffset = -5, yoffset = -7` for `origin57.j2k`, and
     * `extract_area` in this crate stamps `-left` / `-top` for the same
     * meaning, so the two agree on the offsets even where they disagree on the
     * size. Before this the loader attached no offset at all, so a caller had
     * no way to learn the image was not at the grid origin.
     * Input: `origin57.j2k` and the other decodable fixtures -> Output:
     * `-5 / -7` on the first, and `0 / 0` on the rest, matching every
     * `header.xoffset` / `header.yoffset` in `oracle.json`.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn the_image_origin_travels_as_the_negative_offset_vips_records() {
        let raster = decoded("origin57.j2k");
        assert_eq!(
            (raster.xoffset(), raster.yoffset()),
            (-5, -7),
            "the same numbers `vipsheader -a` prints for this file"
        );

        // The positive control, and the reason a sweep rather than one case:
        // every other fixture starts at the grid origin and the capture
        // records `xoffset: 0` for all of them, so a loader that stamped
        // something unconditionally would pass the assertion above. Sweeping
        // `FIXTURES` rather than a hand-written list keeps that true for a
        // fixture added later.
        let mut zeroed = 0;
        for (name, bytes) in FIXTURES {
            if *name == "origin57.j2k" {
                continue;
            }
            let Ok(other) = decode_jp2k(bytes, DecodeLimits::default()) else {
                continue; // the five malformed ones and the three refused carriers
            };
            assert_eq!(
                (other.xoffset(), other.yoffset()),
                (0, 0),
                "{name}: XOsiz and YOsiz are 0 here and vips reports 0 / 0"
            );
            zeroed += 1;
        }
        assert!(
            zeroed >= 18,
            "the sweep has to reach the decodable fixtures, it reached {zeroed}"
        );
    }

    /**
     * The negated origin saturates rather than wrapping, on a `u32` that
     * comes straight off an untrusted `SIZ`.
     * `-XOsiz` does not fit an `i32` above `2^31`, and `XOsiz` is a 32-bit
     * field an attacker writes. Reaching that branch through a real decode
     * needs a file declaring an origin past two billion; testing the
     * arithmetic needs nothing, because it is a private function of a `u32`.
     * That is #706's lesson: its own twin of this helper shipped off by one,
     * saturating at `i32::MIN + 1`, and survived because the fix was
     * "asserted only by reasoning".
     * Input: the four values around the boundary -> Output: exact negation
     * below `2^31`, exactly `i32::MIN` at `2^31`, and `i32::MIN` above it.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn the_negated_origin_saturates_rather_than_wrapping() {
        assert_eq!(negated_origin(0), 0);
        assert_eq!(negated_origin(5), -5);
        assert_eq!(negated_origin(i32::MAX as u32), -i32::MAX);
        // 2^31 negates exactly onto i32::MIN, so this is the one value where
        // "saturate" and "the right answer" are the same number.
        assert_eq!(negated_origin(1 << 31), i32::MIN);
        assert_eq!(negated_origin((1 << 31) + 1), i32::MIN);
        assert_eq!(negated_origin(u32::MAX), i32::MIN);
    }

    // -----------------------------------------------------------------------
    // The container and codestream parsers
    // -----------------------------------------------------------------------

    /**
     * Pins this module's own `SIZ` reader against the same marker segments
     * the capture recorded, for every fixture that has one.
     * The sign bit and the subsampling factors are the reason this parser
     * exists: nothing else in the pipeline reports either, and both decide
     * an answer (refuse, or run the inverse YCC) that a wrong read would
     * silently change.
     * Input: seven fixtures -> Output: the `siz` block
     * `oracle-captures/foreign-jp2k/oracle.json` recorded for each.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn the_siz_reader_reproduces_the_captured_marker_segments() {
        // (fixture, Xsiz, Ysiz, XOsiz, YOsiz, XTsiz, YTsiz, per-component
        // (precision, signed, dx, dy)).
        #[allow(clippy::type_complexity)]
        let cases: [(&str, u32, u32, u32, u32, u32, u32, &[(u8, bool, u8, u8)]); 7] = [
            (
                "rgb_lossless.jp2",
                4,
                3,
                0,
                0,
                512,
                512,
                &[(8, false, 1, 1), (8, false, 1, 1), (8, false, 1, 1)],
            ),
            ("grey_tile8.jp2", 37, 21, 0, 0, 8, 8, &[(8, false, 1, 1)]),
            ("depth12s.j2k", 5, 1, 0, 0, 5, 1, &[(12, true, 1, 1)]),
            ("depth14u.j2k", 5, 1, 0, 0, 5, 1, &[(14, false, 1, 1)]),
            ("int31.jp2", 5, 1, 0, 0, 512, 512, &[(31, true, 1, 1)]),
            ("uint31.jp2", 5, 1, 0, 0, 512, 512, &[(31, false, 1, 1)]),
            (
                "sub420.j2k",
                8,
                4,
                0,
                0,
                8,
                4,
                &[(8, false, 1, 1), (8, false, 2, 2), (8, false, 2, 2)],
            ),
        ];
        for (name, xsiz, ysiz, xo, yo, xt, yt, components) in cases {
            let bytes = fixture(name);
            let layout =
                ContainerLayout::parse(bytes).unwrap_or_else(|e| panic!("{name}: container: {e}"));
            let header = CodestreamHeader::parse(&bytes[layout.codestream..])
                .unwrap_or_else(|e| panic!("{name}: codestream: {e}"));
            assert_eq!(
                (
                    header.xsiz,
                    header.ysiz,
                    header.x_origin,
                    header.y_origin,
                    header.tile_width,
                    header.tile_height
                ),
                (xsiz, ysiz, xo, yo, xt, yt),
                "{name}: SIZ geometry"
            );
            let got: Vec<(u8, bool, u8, u8)> = header
                .components
                .iter()
                .map(|c| (c.precision, c.signed, c.dx, c.dy))
                .collect();
            assert_eq!(got, components.to_vec(), "{name}: SIZ components");
        }
    }

    /**
     * Pins the resolution count read out of the `COD` marker, which is what
     * vips reports as `n-pages`, and the tile grid derived from `SIZ`.
     * `res3.j2k` is the only fixture with more than one resolution, which is
     * exactly why it exists: `jp2ksave`'s formula gives every image small
     * enough to commit a single resolution, so a port that returned a
     * constant 1 would pass on all the others.
     * Input: three fixtures -> Output: 3, 1 and 1 resolutions, and a 5x3,
     * 1x1 and 1x1 tile grid.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn the_cod_marker_carries_the_resolution_count_and_siz_carries_the_tile_grid() {
        for (name, resolutions, grid) in [
            ("res3.j2k", 3, (1, 1)),
            ("grey_tile8.jp2", 1, (5, 3)),
            ("rgb_lossless.jp2", 1, (1, 1)),
        ] {
            let bytes = fixture(name);
            let layout = ContainerLayout::parse(bytes).expect("container");
            let header = CodestreamHeader::parse(&bytes[layout.codestream..]).expect("codestream");
            assert_eq!(header.resolutions, Some(resolutions), "{name}: resolutions");
            assert_eq!(header.tile_grid(), grid, "{name}: tile grid");
        }
    }

    /**
     * Pins the two container shapes apart, which is the whole of the
     * "unspecified colour space" test the inverse YCC hangs off: a JP2
     * always carries a `colr` box and a bare codestream never does, so
     * `bare` is what says whether the colour space is unspecified.
     * Without this, a JP2 whose components happen to be subsampled would
     * have the YCC transform run over pixels the `colr` box already
     * described: `chroma_sub_on.jp2` is exactly that file, and it is the
     * control here.
     * Input: `sub420.j2k` and `chroma_sub_on.jp2` -> Output: subsampled in
     * both, bare in only one.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn only_a_bare_codestream_counts_as_an_unspecified_colour_space() {
        let bare = fixture("sub420.j2k");
        let layout = ContainerLayout::parse(bare).expect("container");
        assert!(layout.bare);
        assert_eq!(layout.codestream, 0);
        let header = CodestreamHeader::parse(&bare[layout.codestream..]).expect("codestream");
        assert!(header.chroma_subsampled());

        // The control, and the reason `bare` is half the condition rather
        // than the subsampling being the whole of it: this fixture is
        // subsampled in exactly the same way and must NOT get the transform,
        // because its colr box already says sYCC and the decoder has already
        // undone it. Running the transform here would apply it twice.
        let boxed = fixture("chroma_sub_on.jp2");
        let layout = ContainerLayout::parse(boxed).expect("container");
        assert!(
            !layout.bare,
            "a JP2 carries a colr box, so its colour space is not unspecified"
        );
        let header = CodestreamHeader::parse(&boxed[layout.codestream..]).expect("codestream");
        assert!(
            header.chroma_subsampled(),
            "the subsampling alone is not what turns the transform on: this fixture has \
             it and must not get it"
        );
    }

    /**
     * Pins the box walker against the shapes that would otherwise loop or
     * run off the end: a length shorter than the header it is part of, a
     * length past the end of the file, and the `LBox = 0` "to the end" form.
     * The walker feeds an offset straight into a slice index, so every one
     * of these is a panic rather than an error if it is not caught, and the
     * decoder eats untrusted bytes.
     * Input: four hand-built box headers -> Output: a typed
     * `Jp2kError::Container` for the three broken ones and a clean walk for
     * the last.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn the_box_walker_refuses_a_length_it_cannot_trust() {
        // A four-byte length of 4 is shorter than the 8-byte header.
        let short = [0u8, 0, 0, 4, b'j', b'p', b'2', b'c'];
        assert!(matches!(
            walk_boxes(&short, 0, short.len()),
            Err(Jp2kError::Container { .. })
        ));

        // A length that runs past the end.
        let over = [0u8, 0, 1, 0, b'j', b'p', b'2', b'c', 1, 2, 3, 4];
        assert!(matches!(
            walk_boxes(&over, 0, over.len()),
            Err(Jp2kError::Container { .. })
        ));

        // An extended length whose XLBox is not there.
        let cut = [0u8, 0, 0, 1, b'j', b'p', b'2', b'c'];
        assert!(matches!(
            walk_boxes(&cut, 0, cut.len()),
            Err(Jp2kError::Container { .. })
        ));

        // The control: `LBox = 0` runs to the end of the enclosing range, and
        // is the one zero this walker accepts.
        let to_end = [0u8, 0, 0, 0, b'j', b'p', b'2', b'c', 9, 9];
        let walked = walk_boxes(&to_end, 0, to_end.len()).expect("LBox 0 is legal");
        assert_eq!(walked.len(), 1);
        assert_eq!((walked[0].start, walked[0].end), (8, 10));
    }

    // -----------------------------------------------------------------------
    // The decode limits
    // -----------------------------------------------------------------------

    /**
     * Pins that all three decode ceilings reach this loader, each with its
     * own typed refusal, and that none of them fires on the same file with
     * the ceiling lifted.
     * The allocation budget is the one worth the fixture: it is priced from
     * the declared geometry before the decoder reserves anything, and it
     * reports through the one shared `SourceError::AllocLimitExceeded` shape
     * every self-pricing decoder in the crate uses (issue #686), carrying
     * the geometry it priced.
     * Input: `grey_tile8.jp2`, 37x21x1 at one byte a sample, so a price of
     * 777 -> Output: a refusal at 776 naming 37x21x1 and 777, and a clean
     * decode with the budget lifted.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn every_decode_ceiling_reaches_this_loader_with_its_own_typed_refusal() {
        use crate::source::DeclaredGeometry;

        let bytes = fixture("grey_tile8.jp2");
        // The control first: the file decodes at the geometry the prices
        // below are computed from.
        let open = DecodeLimits::default().with_max_alloc_bytes(u64::MAX);
        let ok = decode_jp2k(bytes, open).expect("the fixture must decode with limits lifted");
        assert_eq!((ok.width(), ok.height()), (37, 21));

        let err = decode_jp2k(bytes, DecodeLimits::default().with_max_coord(36))
            .expect_err("37 is past a 36-pixel coordinate ceiling");
        assert!(
            matches!(err, SourceError::CoordLimitExceeded { .. }),
            "the coordinate ceiling: {err:?}"
        );

        let err = decode_jp2k(bytes, DecodeLimits::default().with_max_pixels(776))
            .expect_err("777 pixels is past a 776-pixel ceiling");
        assert!(
            matches!(err, SourceError::DimensionLimitExceeded { .. }),
            "the pixel ceiling: {err:?}"
        );

        let err = decode_jp2k(bytes, DecodeLimits::default().with_max_alloc_bytes(776))
            .expect_err("777 bytes is past a 776-byte budget");
        let SourceError::AllocLimitExceeded {
            what,
            geometry,
            needed_bytes,
            max_alloc_bytes,
        } = err
        else {
            panic!("the budget must report the shared shape: {err:?}");
        };
        assert_eq!(what, "JPEG 2000 component buffers");
        assert_eq!(
            geometry,
            Some(DeclaredGeometry {
                width: 37,
                height: 21,
                bands: 1
            }),
            "a decoder that prices a declared geometry reports it"
        );
        assert_eq!((needed_bytes, max_alloc_bytes), (777, 776));
    }

    // -----------------------------------------------------------------------
    // The encoder
    // -----------------------------------------------------------------------

    /// A `width` x `height` raster in `format` whose samples are a
    /// per-band ramp, so a band that ends up in the wrong place is visible.
    #[cfg(feature = "jp2k")]
    fn ramp(width: u32, height: u32, format: PixelFormat) -> Raster {
        let bands = format.channels();
        let wide = format.bytes_per_channel() == 2;
        let mut data =
            Vec::with_capacity(width as usize * height as usize * format.bytes_per_pixel());
        for i in 0..(width as usize * height as usize) {
            for band in 0..bands {
                let value = (i * 7 + band * 40) as u32;
                if wide {
                    data.extend_from_slice(&((value * 271 % 65536) as u16).to_ne_bytes());
                } else {
                    data.push((value % 256) as u8);
                }
            }
        }
        Raster::new(width, height, format, data).expect("ramp fixture")
    }

    /**
     * Pins the lossless encoder as a true round trip at every carrier this
     * codec reads: what goes in comes back out, sample for sample, at 1, 2, 3
     * and 4 bands and at both element widths.
     * The multiband rows are not padding. A 2-band raster is where vips's own
     * band-count guess splits (measured: 2 bands read back `b-w`, 3 read back
     * `srgb`), and a de-interleaver that transposed bands would still
     * round-trip a 1-band image, so the wide rows are what makes the plane
     * ordering an assertion.
     * Input: eight rasters -> Output: the same pixels back, at the same
     * carrier, with the container always a JP2 whatever the raster.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn every_carrier_survives_a_lossless_round_trip_through_this_crate() {
        let two = std::num::NonZeroU16::new(2).expect("2 is non-zero");
        for format in [
            PixelFormat::Gray8,
            PixelFormat::Gray16,
            PixelFormat::Rgb8,
            PixelFormat::Rgb16,
            PixelFormat::Rgba8,
            PixelFormat::Rgba16,
            PixelFormat::Multi8(two),
            PixelFormat::Multi16(two),
        ] {
            let source = ramp(8, 6, format);
            let bytes = source
                .encode_jp2k(SaveOptions::default())
                .unwrap_or_else(|e| panic!("{format:?} must encode: {e}"));
            assert!(
                bytes.starts_with(JP2_SIGNATURE),
                "{format:?}: jp2ksave writes a JP2 container for every suffix it \
                 registers, so this encoder takes no carrier argument at all"
            );
            let back = decode_jp2k(&bytes, DecodeLimits::default())
                .unwrap_or_else(|e| panic!("{format:?} must decode back: {e}"));
            assert_eq!(back.format(), format, "{format:?}: carrier");
            assert_eq!(
                (back.width(), back.height()),
                (8, 6),
                "{format:?}: geometry"
            );
            assert_eq!(
                back.data(),
                source.data(),
                "{format:?}: the reversible 5/3 wavelet and the reversible \
                 multiple-component transform are both exact, so this is an identity"
            );
        }
    }

    /**
     * Pins the lossy mode as actually lossy and actually smaller, and the
     * lossless one as neither.
     * Both halves matter. A `Compression::Lossy` that silently encoded
     * losslessly would still round-trip and would still decode, so "the
     * pixels moved" is the only assertion that can tell the two apart, and
     * "the file shrank" is what says the ratio reached the encoder rather
     * than being accepted and dropped.
     * Input: a 64x64 ramp at ratio 40 -> Output: a smaller file whose
     * pixels differ from the lossless one's.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn a_compression_ratio_reaches_the_encoder_and_costs_accuracy() {
        let ratio = std::num::NonZeroU16::new(40).expect("40 is non-zero");
        let source = ramp(64, 64, PixelFormat::Rgb8);
        let lossless = source
            .encode_jp2k(SaveOptions::default())
            .expect("lossless encode");
        let lossy = source
            .encode_jp2k(SaveOptions {
                compression: Compression::Lossy { ratio },
            })
            .expect("lossy encode");
        assert!(
            lossy.len() < lossless.len(),
            "a 40:1 ratio must reach cp_disto_alloc: {} lossy against {} lossless",
            lossy.len(),
            lossless.len()
        );
        let back = decode_jp2k(&lossy, DecodeLimits::default()).expect("lossy decode");
        assert_ne!(
            back.data(),
            source.data(),
            "the 9/7 wavelet with a rate target is not an identity; if this passes the \
             ratio is being accepted and dropped"
        );
        // The control: the lossless one IS an identity, so the assertion
        // above is about the mode and not about the round trip.
        let exact = decode_jp2k(&lossless, DecodeLimits::default()).expect("lossless decode");
        assert_eq!(exact.data(), source.data());
    }

    /**
     * Pins that a float raster is refused rather than quantised, which is
     * what `vips jp2ksave` does: it fails with `not an integer format`.
     * The control is the same geometry cast to an integer carrier, which
     * must encode, so the refusal is the sample type and not the shape.
     * Input: `FloatF32(3)` and `RgbaF32` -> Output: `EncodeError::Encode`
     * naming the format and telling the caller to cast.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn a_float_raster_is_refused_rather_than_quantised_behind_the_caller() {
        let three = std::num::NonZeroU16::new(3).expect("3 is non-zero");
        for format in [PixelFormat::FloatF32(three), PixelFormat::RgbaF32] {
            let bands = format.channels();
            let data: Vec<u8> = (0..(4 * 3 * bands) as u32)
                .flat_map(|v| (v as f32).to_ne_bytes())
                .collect();
            let raster = Raster::new(4, 3, format, data).expect("float fixture");
            let err = raster
                .encode_jp2k(SaveOptions::default())
                .expect_err("a float raster has no integer samples to write");
            let message = err.to_string();
            assert!(matches!(err, EncodeError::Encode(_)), "{format:?}: {err:?}");
            assert!(
                message.contains("cast to an integer format first"),
                "{format:?}: the message must say what to do: {message}"
            );
        }
        // The control.
        assert!(
            ramp(4, 3, PixelFormat::Rgb8)
                .encode_jp2k(SaveOptions::default())
                .is_ok()
        );
    }

    /**
     * Pins `jp2ksave`'s resolution-count formula, which is the one encoder
     * parameter with an observable answer on the way back out.
     * The ten sizes are the ones the formula was measured over, and four of
     * them (65, 100, 129, 1000) are there because `ceil(log2(n))` gives a
     * different answer at each: without those four rows the test would pass
     * against the wrong formula.
     * Input: ten square sizes -> Output: the resolution count
     * `vips jp2ksave --lossless` put in each file's COD marker.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn the_resolution_count_follows_the_formula_measured_from_jp2ksave() {
        // Measured: `vips black b.v N N`, `vips jp2ksave b.v out.jp2
        // --lossless`, `vipsheader -f n-pages out.jp2`, on 8.18.6.
        let cases: [(u32, i32); 10] = [
            (4, 1),
            (32, 1),
            (64, 1),
            (65, 1),
            (100, 1),
            (128, 2),
            (129, 2),
            (256, 3),
            (512, 4),
            (1000, 4),
        ];
        for (size, want) in cases {
            assert_eq!(
                num_resolutions(size, size),
                want,
                "a {size}x{size} image: floor(log2) is what vips uses; ceil would say {}",
                ((size as f64).log2().ceil() as i32 - 5).max(1)
            );
        }
        // And it reaches the file: a 256x256 encode reads back with three.
        let bytes = ramp(256, 256, PixelFormat::Gray8)
            .encode_jp2k(SaveOptions::default())
            .expect("encode");
        let back = decode_jp2k(&bytes, DecodeLimits::default()).expect("decode");
        assert_eq!(int_field(&back, "jp2k-resolutions"), Some(3));
    }

    /**
     * Pins the `colr` box the encoder writes, which comes from the raster's
     * interpretation and not from its band count: a four-band CMYK raster
     * and a four-band RGBA one are the same `PixelFormat` and want different
     * boxes.
     * Asserted through the round trip rather than through the private
     * helper, because the claim is what a reader sees: a CMYK raster must
     * come back tagged CMYK and an RGBA one must come back tagged sRGB.
     * Input: two four-band `Rgba8` rasters differing only in their tag ->
     * Output: two different interpretations back.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn the_colr_box_follows_the_interpretation_and_not_the_band_count() {
        let mut cmyk = ramp(4, 3, PixelFormat::Rgba8);
        cmyk.meta.interpretation = Some(Interpretation::Cmyk);
        let bytes = cmyk.encode_jp2k(SaveOptions::default()).expect("encode");
        let back = decode_jp2k(&bytes, DecodeLimits::default()).expect("decode");
        assert_eq!(back.interpretation(), Interpretation::Cmyk);
        assert_eq!(back.data(), cmyk.data());

        let rgba = ramp(4, 3, PixelFormat::Rgba8);
        let bytes = rgba.encode_jp2k(SaveOptions::default()).expect("encode");
        let back = decode_jp2k(&bytes, DecodeLimits::default()).expect("decode");
        assert_eq!(
            back.interpretation(),
            Interpretation::Srgb,
            "the same PixelFormat with a different tag must write a different colr box"
        );
    }

    // -----------------------------------------------------------------------
    // Without the feature
    // -----------------------------------------------------------------------

    /**
     * Pins the shape of the build without the `jp2k` feature: every entry
     * point still exists at the same signature and each returns a typed
     * refusal naming the feature, so a caller compiles against either build
     * and can tell "this build has no JPEG 2000" from "these bytes are not
     * JPEG 2000" by the variant rather than by a message.
     * Input: a valid fixture and a raster -> Output:
     * `Jp2kError::FeatureNotEnabled` and `EncodeError::Unsupported`.
     */
    #[test]
    #[cfg(not(feature = "jp2k"))]
    fn without_the_feature_every_entry_point_is_a_typed_refusal() {
        let err = decode_jp2k(fixture("rgb_lossless.jp2"), DecodeLimits::default())
            .expect_err("this build has no JPEG 2000 decoder");
        assert!(
            matches!(err, SourceError::Jp2k(Jp2kError::FeatureNotEnabled)),
            "a build without the feature must say so by variant: {err:?}"
        );

        let raster = Raster::new(4, 3, PixelFormat::Rgb8, vec![7u8; 36]).expect("fixture");
        let err = raster
            .encode_jp2k(SaveOptions::default())
            .expect_err("this build has no JPEG 2000 encoder");
        assert!(
            matches!(err, EncodeError::Unsupported { ref format } if format == "jp2k"),
            "the encode spine reports the format name: {err:?}"
        );
    }

    /**
     * Pins that the sniffer routes both container forms here in either
     * build, which is what makes a JPEG 2000 file report "this build has no
     * JPEG 2000" rather than "these bytes are not an image".
     * The control is the fixture that is not JPEG 2000 at all: it must not
     * route here, which is what vips does too, refusing it in
     * `vips_foreign_find_load` before jp2kload is asked.
     * Input: a JP2, a bare codestream and a file that is neither ->
     * Output: a route for the first two and none for the third.
     */
    #[test]
    fn both_container_forms_route_to_this_loader_and_garbage_does_not() {
        for name in ["rgb_lossless.jp2", "sub420.j2k"] {
            let err = decode_jp2k(fixture(name), DecodeLimits::default());
            // In either build this reaches the loader; with the feature it
            // decodes and without it, it refuses by variant.
            assert_eq!(err.is_ok(), cfg!(feature = "jp2k"), "{name}");
        }
        assert!(
            crate::source::decode_bytes(fixture("not_jp2k.bin")).is_err(),
            "a file that is not an image is refused before any codec is asked"
        );
    }

    /**
     * Pins the band ceiling as a typed refusal with the reason on it, and
     * pins that the ceiling is the *loader's* rather than the format's.
     * `jp2ksave` writes a five-band file and so does this encoder's
     * dependency: measured, one written here reads back through
     * `vips jp2kload` as `5 bands, srgb`, bit for bit. What cannot read it is
     * `decode_jp2k`, so the encoder refuses instead of writing files its own
     * loader rejects (issue #769).
     * The control is four bands, which must encode and round-trip, so the
     * refusal is the count and not the multiband carrier.
     * Input: `Multi8(5)` and `Multi16(5)` -> Output: `EncodeError::Encode`
     * naming both numbers.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn more_bands_than_the_loader_reads_is_refused_on_the_way_out() {
        let five = std::num::NonZeroU16::new(5).expect("5 is non-zero");
        for format in [PixelFormat::Multi8(five), PixelFormat::Multi16(five)] {
            let err = ramp(4, 3, format)
                .encode_jp2k(SaveOptions::default())
                .expect_err("five bands is past the loader's ceiling");
            let message = err.to_string();
            assert!(matches!(err, EncodeError::Encode(_)), "{format:?}: {err:?}");
            assert!(
                message.contains(&MAX_BANDS.to_string()) && message.contains('5'),
                "{format:?}: the refusal must name the ceiling and the count: {message}"
            );
        }
        // The control: the ceiling itself encodes and round-trips.
        let four = ramp(4, 3, PixelFormat::Rgba8);
        let bytes = four
            .encode_jp2k(SaveOptions::default())
            .expect("four bands");
        assert_eq!(
            decode_jp2k(&bytes, DecodeLimits::default())
                .expect("four bands decode")
                .data(),
            four.data()
        );
    }

    /**
     * Pins the other half of the YCC condition, which is that a `colr` box is
     * what makes a JP2's colour space specified, and pins what happens when
     * that box names a value openjpeg does not recognise.
     * Built rather than committed: the same `sub420.j2k` codestream, whose
     * chroma is halved on both axes, wrapped in a minimal JP2 with three
     * different `colr` boxes. That isolates the box from everything else, so
     * the three rows differ in exactly four bytes.
     * Measured on vips 8.18.6, at pixel (0, 0): `EnumCS 16` gives
     * `[128, 16, 240]` and no transform, `EnumCS 18` gives `[255, 87, 0]` and
     * the inverse YCC, and `EnumCS 99` also gives `[255, 87, 0]`, because an
     * unrecognised enum falls through to UNSPECIFIED where the subsampling
     * turns the transform on. libviprs matches the first two and **refuses**
     * the third, because `hayro-jpeg2000` will not parse a `colr` box it does
     * not recognise. That is a refusal rather than a wrong picture, and it is
     * issue #771.
     * Input: three hand-wrapped JP2s -> Output: two decodes matching vips and
     * one typed refusal.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn a_jp2s_colr_box_is_what_makes_its_colour_space_specified() {
        /// Wrap `sub420.j2k`'s codestream in a minimal RFC 3745 JP2 whose
        /// `colr` box is `METH = 1` with `enumcs`.
        fn wrap(enumcs: u32) -> Vec<u8> {
            fn boxed(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
                let mut out = ((payload.len() + 8) as u32).to_be_bytes().to_vec();
                out.extend_from_slice(kind);
                out.extend_from_slice(payload);
                out
            }
            let mut ihdr = Vec::new();
            ihdr.extend_from_slice(&4u32.to_be_bytes()); // height
            ihdr.extend_from_slice(&8u32.to_be_bytes()); // width
            ihdr.extend_from_slice(&3u16.to_be_bytes()); // components
            ihdr.extend_from_slice(&[7, 7, 1, 0]); // bpc, C, UnkC, IPR
            let mut colr = vec![1, 0, 0]; // METH = 1, PREC, APPROX
            colr.extend_from_slice(&enumcs.to_be_bytes());

            let mut out = boxed(b"jP  ", b"\r\n\x87\n");
            let mut ftyp = b"jp2 ".to_vec();
            ftyp.extend_from_slice(&0u32.to_be_bytes());
            ftyp.extend_from_slice(b"jp2 ");
            out.extend_from_slice(&boxed(b"ftyp", &ftyp));
            let mut header = boxed(b"ihdr", &ihdr);
            header.extend_from_slice(&boxed(b"colr", &colr));
            out.extend_from_slice(&boxed(b"jp2h", &header));
            out.extend_from_slice(&boxed(b"jp2c", fixture("sub420.j2k")));
            out
        }

        // EnumCS 16, sRGB: the components are what they are, no transform.
        let srgb = decode_jp2k(&wrap(16), DecodeLimits::default()).expect("sRGB must decode");
        assert_eq!(
            &samples(&srgb)[..3],
            &[128, 16, 240],
            "a specified sRGB colr box leaves the components alone"
        );

        // EnumCS 18, sYCC: the decoder undoes it, so this must NOT get the
        // transform a second time.
        let sycc = decode_jp2k(&wrap(18), DecodeLimits::default()).expect("sYCC must decode");
        let got = &samples(&sycc)[..3];
        for (band, (mine, want)) in got.iter().zip([255u32, 87, 0].iter()).enumerate() {
            assert!(
                mine.abs_diff(*want) <= 1,
                "band {band}: {mine} against vips's {want}; applying the transform twice \
                 would be nowhere near"
            );
        }

        // EnumCS 99, not a defined value: vips reads it and this does not.
        let err = decode_jp2k(&wrap(99), DecodeLimits::default())
            .expect_err("hayro-jpeg2000 will not parse a colr box it does not recognise");
        assert!(
            matches!(err, SourceError::Jp2k(Jp2kError::Decode { .. })),
            "issue #771: this is a refusal and not a wrong picture: {err:?}"
        );
    }

    /**
     * Runs every hand-built malformation in `fuzz/corpus/fuzz_jp2k/` through
     * the real entry point and asserts each one comes back as a typed refusal
     * rather than a panic, an over-read or an unbounded allocation.
     * These are the shapes this module's own parsers have to survive, and it
     * has two of them: an ISO/IEC 15444-1 box walker and a marker-segment
     * walker, both reading attacker-controlled lengths straight into slice
     * indices. Every case below is a length that lies: shorter than its own
     * header, past the end of the file, zero where zero means "to the end",
     * an extended `XLBox` that does not fit in memory, a marker segment
     * shorter than its own length field, a component count past the end of
     * the segment, a subsampling factor of zero that would divide by zero,
     * and a `SIZ` declaring `u32::MAX` on both axes.
     * They are in the corpus so a fuzzer starts from them and here so they are
     * checked on every `cargo test`, because a corpus nobody runs is not a
     * check. The control is the last row: the same 8x4 `SIZ` with nothing
     * wrong, which must get past both walkers and fail in the decoder instead,
     * so a walker that refused everything would not pass this.
     * Input: nineteen hand-built byte strings -> Output: `Err` from every one,
     * and a different error from the control.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn no_hand_built_malformation_panics_or_allocates_without_a_ceiling() {
        const SIG: &[u8] = JP2_SIGNATURE;
        /// A `SIZ` segment body for an 8x4 image with `components` triples
        /// appended by the caller.
        fn siz(components: &[u8], count: u16, geometry: [u32; 8]) -> Vec<u8> {
            let mut out = CODESTREAM_SIGNATURE.to_vec();
            // Saturating, because one case declares 65535 components on
            // purpose and the real length field cannot hold that.
            let lsiz = 38u32 + 3 * u32::from(count);
            out.extend_from_slice(&(lsiz.min(u32::from(u16::MAX)) as u16).to_be_bytes()); // Lsiz
            out.extend_from_slice(&0u16.to_be_bytes()); // Rsiz
            for field in geometry {
                out.extend_from_slice(&field.to_be_bytes());
            }
            out.extend_from_slice(&count.to_be_bytes());
            out.extend_from_slice(components);
            out
        }
        let plain = [8u32, 4, 0, 0, 8, 4, 0, 0];

        let cases: Vec<(&str, Vec<u8>)> = vec![
            ("empty", Vec::new()),
            ("jp2 signature only", SIG.to_vec()),
            (
                "box length under eight",
                [SIG, b"\x00\x00\x00\x04jp2c"].concat(),
            ),
            (
                "box length past the end",
                [SIG, b"\x00\x00\xff\xffjp2c\x01\x02\x03\x04"].concat(),
            ),
            (
                "zero box length that is not last",
                [SIG, b"\x00\x00\x00\x00jp2h", b"\x00\x00\x00\x08jp2c"].concat(),
            ),
            (
                "extended box length truncated",
                [SIG, b"\x00\x00\x00\x01jp2c"].concat(),
            ),
            (
                "extended box length past usize",
                [SIG, b"\x00\x00\x00\x01jp2c\xff\xff\xff\xff\xff\xff\xff\xff"].concat(),
            ),
            (
                "jp2h child runs past its parent",
                [
                    SIG,
                    b"\x00\x00\x00\x13jp2h",
                    b"\x00\x00\x00\xffcolr\x02\x00\x00",
                ]
                .concat(),
            ),
            ("soc without siz", b"\xff\x4f\xff\x52\x00\x02".to_vec()),
            (
                "siz truncated after the marker",
                CODESTREAM_SIGNATURE.to_vec(),
            ),
            ("zero components", siz(&[], 0, plain)),
            (
                "component count past the end of the segment",
                siz(&[7, 1, 1], 0xffff, plain),
            ),
            ("zero subsampling factor", siz(&[7, 0, 0], 1, plain)),
            (
                "u32::MAX on both axes",
                siz(
                    &[7, 1, 1],
                    1,
                    [u32::MAX, u32::MAX, 0, 0, u32::MAX, u32::MAX, 0, 0],
                ),
            ),
            (
                "marker segment length under two",
                [siz(&[7, 1, 1], 1, plain), b"\xff\x52\x00\x00".to_vec()].concat(),
            ),
            (
                "cod segment truncated",
                [
                    siz(&[7, 1, 1], 1, plain),
                    b"\xff\x52\x00\x05\x00\x00\x00".to_vec(),
                ]
                .concat(),
            ),
            (
                "not a marker in the main header",
                [siz(&[7, 1, 1], 1, plain), b"\x00\x00\x00\x04".to_vec()].concat(),
            ),
            (
                "tile origin past the image",
                siz(&[7, 1, 1], 1, [8, 4, 0, 0, 1, 1, 99, 99]),
            ),
        ];

        // A very small budget, so an unbounded allocation shows up as a hang
        // or an abort rather than as a pass.
        let limits = DecodeLimits::default().with_max_alloc_bytes(64 * 1024);
        for (name, bytes) in &cases {
            let err = decode_jp2k(bytes, limits)
                .err()
                .unwrap_or_else(|| panic!("{name}: a malformed input must not decode"));
            assert!(
                matches!(
                    err,
                    SourceError::Jp2k(_) | SourceError::AllocLimitExceeded { .. }
                ),
                "{name}: must refuse in a typed shape: {err:?}"
            );
        }

        // The control: the same `SIZ` with nothing wrong gets past both
        // walkers and fails inside the decoder instead, which is what says the
        // walkers are refusing specific things rather than everything.
        let well_formed = siz(&[7, 1, 1], 1, plain);
        let err = decode_jp2k(&well_formed, limits)
            .expect_err("a header with no tile data behind it still cannot decode");
        assert!(
            matches!(err, SourceError::Jp2k(Jp2kError::Decode { .. })),
            "a well-formed header must reach the decoder rather than the walkers: {err:?}"
        );
    }

    /**
     * Pins that the precision ceiling is checked BEFORE the frame is priced,
     * which is the half of that guard nothing else could see.
     * There are two precision checks and they are two guards rather than one
     * written twice. The first reads the precision out of `SIZ`, before the
     * allocation budget is priced and before the decoder is called at all. The
     * second reads what the decoder actually returned, because a palette
     * declares one component in `SIZ` and expands to three, which `SIZ` cannot
     * predict. Mutation testing found that deleting the first left
     * `a_carrier_this_crate_does_not_have_is_a_typed_refusal_and_not_an_offset`
     * green, because the second catches the same file and reports the same
     * variant, so that test says nothing about the order.
     * The order is what this pins, and it is observable: `uint31.jp2` is 5x1
     * with one 31-bit component, so a build that got past the `SIZ` check
     * would price it at two bytes a sample and ten bytes in total, and under a
     * nine-byte budget the two answers are different variants.
     * The control is `depth16u.j2k`, the same 5x1 geometry at 16 bits, which
     * passes the precision check and must then trip the budget: without it,
     * "the precision error wins" would pass for a build where the budget was
     * simply unreachable.
     * Input: `uint31.jp2` under `max_alloc_bytes = 9` -> Output:
     * `PrecisionNotSupported`, where the control gives `AllocLimitExceeded`.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn the_precision_ceiling_is_checked_before_the_frame_is_priced() {
        let tight = DecodeLimits::default().with_max_alloc_bytes(9);

        let err = decode_jp2k(fixture("uint31.jp2"), tight)
            .expect_err("31 bits per sample has no carrier");
        assert!(
            matches!(
                err,
                SourceError::Jp2k(Jp2kError::PrecisionNotSupported { .. })
            ),
            "the SIZ-level ceiling must be reached before the frame is priced, so it \
             beats the allocation budget rather than the other way round: {err:?}"
        );

        let err = decode_jp2k(fixture("depth16u.j2k"), tight)
            .expect_err("a 5x1 16-bit frame is ten bytes and the budget is nine");
        assert!(
            matches!(err, SourceError::AllocLimitExceeded { .. }),
            "the control must actually reach the budget, or the assertion above is \
             about a ceiling nothing could have hit: {err:?}"
        );
    }

    /**
     * Pins the palette, which is the one file shape where the codestream
     * header and the decoded components legitimately disagree, and the only
     * thing that can reach
     * [`Jp2kError::PrecisionWiderThanDeclared`](Jp2kError::PrecisionWiderThanDeclared).
     * A palettised codestream declares ONE component in `SIZ`, an index, and
     * the `pclr` box maps it to as many columns as it likes at whatever
     * precision it likes. That is why the band count in `decode_jp2k` comes
     * off the decoder rather than off `SIZ`, and this is the file that says
     * so: `SIZ` has `Csiz = 1` and the raster has three bands.
     * Built rather than committed, from the same 5x1 8-bit codestream the
     * depth sweep uses, so the two rows differ in exactly the palette
     * precision.
     * Measured on vips 8.18.6: the 8-bit palette decodes to
     * `5x1 uchar, 3 bands, srgb` with `[0, 0, 0]`, `[1, 2, 3]` and
     * `[64, 128, 192]` at the first three pixels, and the 16-bit one reports
     * the same header and then fails every pixel read with `error in tile 0`.
     * So refusing the second is not a divergence, and the first is a real
     * decode this loader has to get right.
     * Input: two hand-built palettised JP2s -> Output: three bands of vips's
     * pixels from the first and a typed refusal naming both widths from the
     * second.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn a_palette_is_where_siz_and_the_decoded_components_legitimately_disagree() {
        /// Wrap `depth8u.j2k` in a JP2 with a 256-entry, 3-column palette
        /// whose entries are `depth` bits wide.
        fn palettised(depth: u8) -> Vec<u8> {
            fn boxed(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
                let mut out = ((payload.len() + 8) as u32).to_be_bytes().to_vec();
                out.extend_from_slice(kind);
                out.extend_from_slice(payload);
                out
            }
            let (entries, columns) = (256usize, 3usize);
            let width = usize::from(depth) / 8;
            let mut pclr = (entries as u16).to_be_bytes().to_vec();
            pclr.push(columns as u8);
            pclr.extend(std::iter::repeat_n(depth - 1, columns)); // Bi
            for i in 0..entries {
                for column in 0..columns {
                    let value = (i * (column + 1) * 257) % (1usize << depth);
                    pclr.extend_from_slice(&value.to_be_bytes()[8 - width..]);
                }
            }
            let mut cmap = Vec::new();
            for column in 0..columns {
                cmap.extend_from_slice(&0u16.to_be_bytes()); // CMP: component 0
                cmap.push(1); // MTYP: palette mapping
                cmap.push(column as u8); // PCOL
            }
            let mut ihdr = Vec::new();
            ihdr.extend_from_slice(&1u32.to_be_bytes()); // height
            ihdr.extend_from_slice(&5u32.to_be_bytes()); // width
            ihdr.extend_from_slice(&1u16.to_be_bytes()); // components
            ihdr.extend_from_slice(&[7, 7, 1, 0]);
            let mut colr = vec![1, 0, 0];
            colr.extend_from_slice(&16u32.to_be_bytes()); // EnumCS 16, sRGB

            let mut header = boxed(b"ihdr", &ihdr);
            header.extend_from_slice(&boxed(b"colr", &colr));
            header.extend_from_slice(&boxed(b"pclr", &pclr));
            header.extend_from_slice(&boxed(b"cmap", &cmap));

            let mut out = boxed(b"jP  ", b"\r\n\x87\n");
            let mut ftyp = b"jp2 ".to_vec();
            ftyp.extend_from_slice(&0u32.to_be_bytes());
            ftyp.extend_from_slice(b"jp2 ");
            out.extend_from_slice(&boxed(b"ftyp", &ftyp));
            out.extend_from_slice(&boxed(b"jp2h", &header));
            out.extend_from_slice(&boxed(b"jp2c", fixture("depth8u.j2k")));
            out
        }

        // The 8-bit palette: one component in SIZ, three bands out, and vips's
        // pixels.
        let bytes = palettised(8);
        let layout = ContainerLayout::parse(&bytes).expect("container");
        let header = CodestreamHeader::parse(&bytes[layout.codestream..]).expect("codestream");
        assert_eq!(
            header.components.len(),
            1,
            "the codestream carries one component, the palette index"
        );

        let raster = decode_jp2k(&bytes, DecodeLimits::default()).expect("an 8-bit palette");
        assert_eq!(raster.format(), PixelFormat::Rgb8);
        assert_eq!(
            &samples(&raster)[..9],
            &[0, 0, 0, 1, 2, 3, 64, 128, 192],
            "the band count comes off the decoder because SIZ cannot see the palette"
        );

        // The 16-bit palette: wider than the index SIZ declared, so the
        // carrier the frame was priced for cannot hold it.
        let err = decode_jp2k(&palettised(16), DecodeLimits::default())
            .expect_err("16-bit palette entries do not fit the 8-bit carrier SIZ priced");
        let SourceError::Jp2k(Jp2kError::PrecisionWiderThanDeclared {
            component,
            declared,
            decoded,
        }) = err
        else {
            panic!("the refusal must name both widths rather than the ceiling: {err:?}");
        };
        assert_eq!((component, declared, decoded), (0, 8, 16));
    }
}
