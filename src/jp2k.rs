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
//! | [`Raster::encode_jp2k`] | `jp2ksave_buffer` | `.jp2` bytes, always in a JP2 container, tiled on `jp2ksave`'s own 512x512 grid by default |
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
//!   fixtures in the decodable set, **fifteen are byte-identical** to what
//!   `vips rawsave` wrote: every reversible 5/3 file, at every component
//!   precision from 2 to 16 bits, signed and unsigned, greyscale, RGB, RGBA
//!   and CMYK, tiled and untiled, subsampled and multi-resolution. The
//!   reversible wavelet is integer-specified, so that is a parity port
//!   rather than an approximation and its pins carry no tolerance at all.
//!   Of the other seven, four are the irreversible fixtures below, two are
//!   refused on carrier grounds (both of them 31-bit), and one is
//!   `origin57.j2k`, whose geometry diverges and is issue #766.
//!   `depth12s.j2k` moved into the first group when issue #905 landed the
//!   signed carriers.
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
//! * **Signed components go both ways, and the sign bit is the file's own.**
//!   JPEG 2000 carries a per-component `sgnd` flag and vips round-trips it
//!   exactly, so [`PixelFormat::Int8`] and [`PixelFormat::Int16`] save and
//!   load rather than being refused (issue #905). Measured with
//!   `--lossless` on a raster holding `[-5, 100, -100, 7]`: `char` and
//!   `short` come back unchanged, where `int` comes back `[-10, 200, -200,
//!   14]` and `uint` comes back offset by 2^31. The encoder sets `Ssiz`'s
//!   top bit exactly as `jp2ksave` does (`0x87` for an 8-bit signed
//!   component, `0x8f` for a 16-bit one) and `vips jp2kload` reads the
//!   files it writes back sample for sample. The loader takes the sign bit
//!   off the codestream's own `SIZ` marker, because `hayro-jpeg2000` does
//!   not report signedness and hands every component back DC-level-shifted
//!   into the unsigned range, so the file's own sample is what the decoder
//!   returns minus `2^(precision - 1)`. Measured on `depth12s.j2k`:
//!   `hayro-jpeg2000` says `[0, 2047, 2048, 2049, 4095]`, the file holds
//!   `[-2048, -1, 0, 1, 2047]`, and vips says `[-32768, -16, 0, 16, 32752]`
//!   after the left-justification below.
//! * **Two signed shapes are refused, and neither of them is a carrier
//!   gap.** Components that *disagree* on the sign bit have no single
//!   raster carrier, and `vips jp2kload` refuses the same file
//!   (`components differ in precision`, measured on `rgb_lossless.jp2` with
//!   component 1's `Ssiz` bit flipped), so
//!   [`Jp2kError::MixedComponentSignedness`] is parity rather than a gap. A
//!   signed file in the inverse-YCC shape below is refused too, and there
//!   vips does answer: it runs the transform with the offset subtraction
//!   wrapping in the component's own signed carrier and then clamps the
//!   result to the *unsigned* range before storing it back into a `char`.
//!   Measured on the committed `sub420.j2k` shape written signed, the red
//!   band comes out 0 at every pixel and the blue band wraps past 127 into
//!   negatives (`[0, 5, 28]` at pixel 0, `[0, 28, -122]` at pixel 4). That
//!   is an answer no carrier can hold, so [`Jp2kError::SignedInverseYcc`]
//!   refuses instead of reproducing it.
//! * **More than 16 bits of precision is refused, and the reason is the
//!   decoder rather than the carrier.** [`PixelFormat`] grew 32-bit integer
//!   carriers in issue #516, but `hayro-jpeg2000` hands samples back as
//!   `f32`, whose 24-bit mantissa cannot hold a 31-bit sample: measured on
//!   `int31.jp2`, three distinct input values all come back as the same
//!   float. vips's own answer there is not a round trip either (`jp2ksave`
//!   writes 31 bits for a 4-byte format and `jp2kload` doubles them coming
//!   back), so there is nothing to be faithful to.
//!   [`Jp2kError::PrecisionNotSupported`] names the ceiling.
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
//! * **The save is tiled, and it is tiled by default.** `jp2ksave` sets
//!   `tile_size_on` unconditionally with `--tile-width` / `--tile-height`
//!   defaulting to 512, so vips cuts up anything larger than that: measured,
//!   a 600x600 image it writes reports `tile-width: 512`. `Encoder::encode`
//!   writes one tile and `openjpeg2-pure-rs` has no knob for the grid, so
//!   `encode` reaches it through the format instead, encoding each tile as
//!   a standalone image placed at the tile's absolute grid coordinates and
//!   splicing the tile-parts under one main header. JPEG 2000 codes tiles
//!   independently, which is what makes that a rearrangement rather than a
//!   re-encode, and byte identity with the oracle is what says so: over 800
//!   combinations of image size, tile grid, band count and sample depth, 770
//!   are byte-identical to `vips jp2ksave`'s codestream and the other 30 are
//!   the one-pixel-wide-tile rows, which this encoder and vips both refuse
//!   with the same complaint about the resolution count. Issue #768.
//! * **An alpha channel is labelled, on the two shapes vips labels one on.**
//!   `jp2ksave` writes a `cdef` box for greyscale plus a band and for RGB plus
//!   a band, and for nothing else. That is narrower than
//!   `vips_image_hasalpha`, which is true for two bands under any tag and for
//!   anything past four: measured over six band counts and six
//!   interpretations, CMYK plus a band gets no box and neither does a
//!   five-band image. Issue #935.
//! * **Any band count the format carries.** The ceiling used to be four, and
//!   the reason was `hayro-jpeg2000` refusing a component set it cannot map
//!   onto greyscale, RGB, CMYK or one of those plus alpha. That turned out to
//!   be a property of the `colr` box: with no colour specification the
//!   decoder answers `ColorSpace::Unknown { num_channels }` for any count,
//!   which is the arm this module already handled and exactly vips's own
//!   guess. So a file the decoder refuses on the channel count is handed to
//!   it again with the box removed (a bare codestream is wrapped in a
//!   container with an empty `jp2h` instead, since the decoder synthesises an
//!   sRGB box for the bare form before it validates anything), and the
//!   encoder stops where `Csiz` does. Measured against vips on 5, 6 and 8
//!   bands: `N bands, srgb`, every sample identical. Issue #769.
//! * **The `colr` box decides the inverse YCC, and `SIZ` decides it only when
//!   the box does not.** The transform runs when the resolved colour space is
//!   sYCC or e-YCC, and "resolved" means the box's enum where openjpeg
//!   recognises one and `opj_j2k_read_siz`'s heuristic (three components with
//!   the chroma pair subsampled) where it does not. Measured on 8.18.6 over
//!   `chroma_sub_on.jp2`, whose chroma is halved on both axes, by rewriting
//!   nothing but its `colr` box:
//!
//!   | box | `vips getpoint 0 0` | transform |
//!   |---|---|---|
//!   | `METH = 1`, sYCC, as committed | `4 1 241` | yes |
//!   | `METH = 1`, sRGB | `29 248 110` | no |
//!   | `METH = 2`, a profile | `4 1 241` | yes |
//!
//!   So a recognised non-YCC enum suppresses the heuristic and a profile box
//!   does not, because a profile leaves the colour space exactly where `SIZ`
//!   put it. This module asked `bare && subsampled` until #771, which is the
//!   same answer as the rule above on every file whose enum the decoder
//!   handled, and the wrong one on the third row.
//! * **The interpretation comes from the `colr` box, not the band count.**
//!   `jp2kload` reads the box's `EnumCS` and maps openjpeg's five recognised
//!   values onto a tag; anything else falls through to UNSPECIFIED, where it
//!   guesses from the component count. This module does the same, and the
//!   two rows that make it a rule rather than a coincidence are the ones where
//!   the enum and the band count disagree: a **one**-component file tagged
//!   CMYK is `cmyk`, and a **three**-component file tagged greyscale is `b-w`.
//!   The element width picks between the flavours, so the same enum gives
//!   `b-w` / `srgb` on an 8-bit file and `grey16` / `rgb16` on a 16-bit one.
//!
//!   One combination is outright broken in vips and is deliberately not
//!   reproduced: a one-component file tagged sRGB, sYCC or e-YCC has its
//!   header expanded to 3 bands by openjpeg while the tile decode still yields
//!   1, so `vipsheader` reports `3 bands, srgb` and **any pixel read fails**.
//!   This keeps the one real band and takes vips's tag, which is the half of
//!   its answer that is not broken. The tag and the band count are independent
//!   here, as [`Interpretation`] says
//!   outright, so that costs nothing. Issue #767.
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
//! * **The `colr` box does not reach the decoder unless the decoder gets it
//!   right.** `hayro-jpeg2000` resolves the box itself, and on three kinds of
//!   value that is a refusal or a conversion where `jp2kload` reads the file
//!   and leaves the samples alone: an enum openjpeg does not recognise (#771),
//!   e-YCC (#848), and CIELab, which also converts the pixels on the shapes
//!   that do decode (#849). None of it is a property of the codestream, which
//!   decodes perfectly well, so the enum is rewritten to sRGB on the way in
//!   and this module keeps every decision the box makes: the interpretation,
//!   and the inverse YCC.
//!
//!   Only the boxes the decoder gets wrong are rewritten. CMYK, sRGB,
//!   greyscale and sYCC-over-three-components go through untouched, each of
//!   them pinned by a committed fixture, so no file that decoded before this
//!   hands the decoder different bytes.

use std::num::NonZeroU32;
use std::path::Path;

use thiserror::Error;

use crate::codec::EncodeError;
#[cfg(feature = "jp2k")]
use crate::conversion::Interpretation;
#[cfg(feature = "jp2k")]
use crate::imageio::MetadataValue;
use crate::imageio::SaveError;
#[cfg(feature = "jp2k")]
use crate::pixel::{PixelFormat, SampleKind};
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
/// This is the **format's** ceiling now, `Csiz`'s range in ISO/IEC 15444-1
/// Table A.9, and it is measured rather than read off the table:
/// `openjpeg2-pure-rs` encodes 16384 components and refuses 16385 with
/// `EncoderSetupFailed`.
///
/// It used to be 4, and the reason was the loader. `hayro-jpeg2000` refuses a
/// component set it cannot map onto greyscale, RGB, CMYK or one of those plus
/// alpha, so a wider file was one this crate could write and not read back.
/// That turned out to be a property of the `colr` box rather than of the
/// decoder: with no colour specification at all it answers
/// `ColorSpace::Unknown { num_channels }` for any count, which is the arm
/// this module's `interpretation` already handled and exactly vips's own
/// guess. `ContainerLayout::unspecified_rewrite` is what hands it that file, and
/// issue #769 has the sweep.
pub const MAX_BANDS: usize = 16384;

/// The highest component precision [`decode_jp2k`] will carry.
///
/// The ceiling is the decoder's, not the carrier's. [`PixelFormat`] grew
/// [`PixelFormat::Uint32`] and [`PixelFormat::Int32`] in issue #516, so
/// there is somewhere to put a wider sample now; what there is not is a way
/// to get one out of `hayro-jpeg2000`, which returns samples as `f32` and
/// whose 24-bit mantissa cannot hold a 31-bit one. Measured on `int31.jp2`,
/// whose five distinct 31-bit samples come back as three distinct floats.
/// vips does not round-trip 31-bit samples either, so there is nothing on
/// the other side of the ceiling to be faithful to (issue #905 records the
/// numbers).
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
    /// The components disagree about the sign bit, so there is no single
    /// carrier for the raster.
    ///
    /// A [`PixelFormat`] has one sample kind for the whole image, and JPEG
    /// 2000 sets `sgnd` per component, so a file that mixes them has no
    /// carrier at all rather than an awkward one. `vips jp2kload` refuses
    /// the same file: measured on `rgb_lossless.jp2` with component 1's
    /// `Ssiz` sign bit flipped and nothing else touched, vips fails with
    /// `components differ in precision`, while the untouched file decodes.
    /// So this is parity and not a gap.
    #[error(
        "jp2k: component {signed} is signed and component {unsigned} is not; a raster \
         carries one sample kind for every band, and vips jp2kload refuses a file whose \
         components disagree here too"
    )]
    MixedComponentSignedness {
        /// The first signed component, counting from zero.
        signed: usize,
        /// The first unsigned component, counting from zero.
        unsigned: usize,
    },
    /// A signed file is in the shape `jp2kload` runs its inverse YCC over,
    /// and vips's answer there is not one a carrier can hold.
    ///
    /// The shape is the one the module docs describe: a bare or unspecified
    /// colour space, three components, and subsampling on components 1 and
    /// 2. vips does answer for it, and the answer is unusable. Measured on
    /// the committed `sub420.j2k` written signed
    /// instead of unsigned, vips subtracts the YCC offset **inside the
    /// component's own signed carrier**, so `-112 - 128` wraps to `16`, and
    /// then clamps the transform's result to the *unsigned* range before
    /// storing it into a `char`: the red band is 0 at every pixel and the
    /// blue band wraps past 127 into negatives (`[0, 5, 28]` at pixel 0 and
    /// `[0, 28, -122]` at pixel 4, against `[255, 87, 0]` and `[200, 109,
    /// 36]` for the same file written unsigned).
    ///
    /// Reproducing that would be matching an oracle that has lost the
    /// picture, so this refuses instead. The encoder here cannot produce
    /// the shape: it always writes a JP2 container with an explicit `colr`
    /// box and never subsamples, so only a hand-built codestream reaches
    /// this.
    #[error(
        "jp2k: this is a signed {components}-component codestream with subsampled chroma \
         and no declared colour space, which is the shape jp2kload runs its inverse YCC \
         over; vips clamps that transform into the unsigned range and stores it in a \
         signed carrier, losing the picture, so it is refused rather than reproduced"
    )]
    SignedInverseYcc {
        /// How many components the codestream declared, which the shape
        /// fixes at three.
        components: usize,
    },
    /// A component declares more bits per sample than this loader carries.
    ///
    /// The ceiling is [`MAX_PRECISION`] and the reason is on it.
    #[error(
        "jp2k: component {component} declares {precision} bits per sample; this loader \
         carries at most {max}, because the decoder returns samples as f32 and a 24-bit \
         mantissa cannot hold a wider one"
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
/// (measured). There is no `subsample_mode` either, because
/// `openjpeg2-pure-rs`'s `EncodeOptions` exposes `format`, `threads`,
/// `irreversible`, `use_mct`, `rates` and `num_resolutions` and nothing else.
///
/// The tile geometry is here anyway, and **not** through that struct. Two
/// earlier readings of why it could not be were both wrong, so both are
/// recorded rather than quietly replaced. #768 said `cp_tdx` / `cp_tdy` and
/// `tile_size_on` are `pub(crate)`; they are `pub`, on a `pub` struct. The
/// correction said they therefore live in `pub mod openjpeg` and are
/// reachable; the **module** is `pub(crate)`, which a compile probe against
/// 0.1.1 settles in one line:
///
/// ```text
/// error[E0603]: module `openjpeg` is private
///  --> openjpeg2-pure-rs-0.1.1/src/lib.rs:72:1
/// ```
///
/// So `Encoder::encode` really does write one tile and there really is no
/// knob. What there is instead is the format: JPEG 2000 codes every tile
/// independently, so the tile-part a tiled codestream carries for a tile is
/// the tile-part a standalone codestream carries for the same region placed
/// at the same absolute grid coordinates. `encode` encodes each tile that
/// way and splices the parts under one main header, and the evidence that
/// this is the same thing OpenJPEG does is byte identity with the oracle:
/// over 800 combinations of image size, tile grid, band count and sample
/// depth, **770 are byte-identical to the codestream `vips jp2ksave` writes**
/// and the other 30 are the tile-width-1 rows, which this encoder and vips
/// both refuse. Issue #768.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct SaveOptions {
    /// How to compress. Defaults to [`Compression::Lossless`].
    pub compression: Compression,
    /// Tile width, `jp2ksave --tile-width`. Defaults to
    /// [`DEFAULT_TILE_SIZE`].
    pub tile_width: NonZeroU32,
    /// Tile height, `jp2ksave --tile-height`. Defaults to
    /// [`DEFAULT_TILE_SIZE`].
    pub tile_height: NonZeroU32,
}

impl Default for SaveOptions {
    fn default() -> Self {
        Self {
            compression: Compression::Lossless,
            tile_width: DEFAULT_TILE_SIZE,
            tile_height: DEFAULT_TILE_SIZE,
        }
    }
}

impl SaveOptions {
    /// Set the compression mode, returning the updated options.
    #[must_use]
    pub fn with_compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    /// Set the tile width, returning the updated options.
    #[must_use]
    pub fn with_tile_width(mut self, tile_width: NonZeroU32) -> Self {
        self.tile_width = tile_width;
        self
    }

    /// Set the tile height, returning the updated options.
    #[must_use]
    pub fn with_tile_height(mut self, tile_height: NonZeroU32) -> Self {
        self.tile_height = tile_height;
        self
    }
}

/// `jp2ksave`'s own default tile size, on both axes.
///
/// It is 512 rather than "the whole image", which is worth stating because it
/// means vips tiles **by default**: measured, `vips jp2ksave` on a 600x600
/// image writes a file `vipsheader` reports as `tile-width: 512`. A port that
/// writes one tile matches vips on small images and diverges on every large
/// one, which is the half of #768 the issue does not mention.
///
/// The type is [`NonZeroU32`] because vips's own property minimum is 1:
/// `--tile-width 0` is refused by GObject with "value 0 ... is invalid or out
/// of range for property 'tile-width'" and the default is used instead, so
/// there is no zero to be faithful to.
pub const DEFAULT_TILE_SIZE: NonZeroU32 = NonZeroU32::new(512).expect("512 is not zero");

/// The most tiles a codestream can carry, because `Isot` is two bytes.
///
/// Measured rather than read off Table A.5: `vips jp2ksave` on a 256x256
/// image at one pixel per tile fails with "Invalid number of tiles : 256 x 256
/// (maximum fixed by jpeg2000 norm is 65535 tiles)", which is OpenJPEG's own
/// message, and 255x255 gets past that check.
#[cfg(feature = "jp2k")]
const MAX_TILES: u64 = 65535;

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
/// * [`Jp2kError::PrecisionNotSupported`] for a component above
///   [`MAX_PRECISION`], which is the decoder's `f32` container rather than a
///   format limit, and [`Jp2kError::PrecisionWiderThanDeclared`] for a
///   palette whose entries are wider than the index `SIZ` declared.
/// * [`Jp2kError::MixedComponentSignedness`] when the components disagree
///   about the sign bit, and [`Jp2kError::SignedInverseYcc`] for a signed
///   file in the shape `jp2kload` runs its inverse YCC over. A signed
///   component on its own is a carrier and not a refusal (issue #905).
/// * [`Jp2kError::UnsupportedBandCount`],
///   [`Jp2kError::BandCountMismatch`] and
///   [`Jp2kError::ComponentGeometryMismatch`], all defensive.
/// * [`Jp2kError::Raster`] when the decoded frame cannot be wrapped.
/// * [`SourceError::CoordLimitExceeded`] when either declared axis exceeds
///   [`DecodeLimits::max_coord`], [`SourceError::DimensionLimitExceeded`] when
///   `width * height` exceeds `max_pixels`, and
///   [`SourceError::AllocLimitExceeded`] when the component buffers the header
///   declares would exceed `max_alloc_bytes`. That price covers what
///   `hayro-jpeg2000` holds beside the raster as well as the raster itself,
///   because `max_alloc_bytes` is a ceiling on peak memory and this decoder
///   was measured at 6.45x a raster-only price (issue #944). It is therefore
///   stricter than it was, by roughly an order of magnitude for a small
///   image and by about five times for a large one.
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
    /// [`PixelFormat::Rgba8`], [`PixelFormat::Rgba16`] and the
    /// [`PixelFormat::Multi8`] / [`PixelFormat::Multi16`] forms at any other
    /// count. The float carriers are refused rather than cast, which is what
    /// `vips jp2ksave` does with a `float` or `double` image, and so is
    /// anything wider than [`MAX_BANDS`], which is the codestream's own
    /// `Csiz` range.
    ///
    /// The output is tiled on [`SaveOptions`]' grid, which defaults to
    /// `jp2ksave`'s 512x512 and so cuts up anything larger than that, and it
    /// is always a JP2 container, never a bare codestream, because
    /// `jp2ksave` hard-codes `OPJ_CODEC_JP2` and writes the same bytes for all
    /// five suffixes it registers. Nothing attached to the raster is written:
    /// no ICC profile, no EXIF, no XMP, because `jp2ksave.c` has no code for
    /// any of them.
    ///
    /// The one thing the raster's [`crate::conversion::Interpretation`] does
    /// reach is the `cdef` box, which says which channel is the alpha.
    /// `jp2ksave` writes one for greyscale plus a band and for RGB plus a
    /// band and for nothing else, and so does this.
    ///
    /// # Errors
    ///
    /// [`EncodeError::Unsupported`] naming `"jp2k"` when the crate was built
    /// without the `jp2k` feature, which is the variant every format without
    /// an encoder in this build reports; otherwise [`EncodeError::Encode`]
    /// when the raster is float (cast first; the message says so), when it
    /// has more than [`MAX_BANDS`] bands, when the tile grid is more than
    /// 65535 tiles, or when the codec rejects the frame (a tile too small for
    /// the resolution count is the one to expect, and `vips jp2ksave` refuses
    /// the same grid).
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
/// four things it needs and cannot get otherwise: the `colr` box's enumerated
/// colour space (which decides the interpretation and the inverse YCC, and
/// which the decoder must not be allowed to resolve itself), the `METH=2` ICC
/// payload (which the decoder
/// drops when it cannot parse it, and `icc_colr.jp2` deliberately cannot be
/// parsed), the per-component sign bit and subsampling factors, and the tile
/// geometry.
#[cfg(feature = "jp2k")]
#[derive(Clone, Debug, PartialEq, Eq)]
struct ContainerLayout {
    /// Offset of the `SOC` marker opening the contiguous codestream.
    codestream: usize,
    /// The payload of a `METH=2` `colr` box, which is an ICC profile, copied
    /// verbatim and unvalidated the way `jp2kload` copies it.
    icc: Option<Vec<u8>>,
    /// Byte offset of the four `EnumCS` bytes inside the file, for the
    /// neutralised copy [`decode`] hands the decoder. `None` whenever
    /// [`ContainerLayout::enum_cs`] is `None`.
    enum_cs_at: Option<usize>,
    /// The `EnumCS` of a `METH=1` `colr` box, which is what decides the
    /// interpretation for vips and now for this module too (#767).
    ///
    /// `None` for a bare codestream, which has no `colr` box at all, and for a
    /// `METH=2` box, which carries a profile instead. Both fall back to the
    /// decoder's resolved colour space, which is where they were before.
    enum_cs: Option<u32>,
    /// Whether the file is a bare codestream rather than a JP2 container.
    ///
    /// A bare codestream has no boxes at all, so there is no `colr` box to
    /// drop and [`ContainerLayout::unspecified_rewrite`] wraps it instead.
    bare: bool,
    /// Where the `colr` box is, and which `jp2h` holds it, for the copy
    /// [`ContainerLayout::unspecified_rewrite`] makes.
    ///
    /// `None` for a bare codestream, for a JP2 with no `colr` box (which
    /// already reaches the decoder's unspecified arm), and for the one
    /// container shape the removal cannot keep consistent, a `jp2h` whose
    /// length is not a plain 32-bit `LBox`.
    colr: Option<ColrPlacement>,
}

/// Where a `colr` box sits, and enough about its parent to remove it.
///
/// Whole-box ranges, `LBox` included, which is what a removal has to splice
/// out. [`BoxRef`] carries the same convention in `at` and `end`.
#[cfg(feature = "jp2k")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ColrPlacement {
    /// First byte of the `colr` box.
    at: usize,
    /// One past its last byte.
    end: usize,
    /// First byte of the `jp2h` box holding it, which is the first byte of
    /// the `LBox` the removal rewrites.
    header_at: usize,
    /// One past the `jp2h` box's last byte.
    header_end: usize,
}

/// One top-level or sub-box, as an offset range into the file.
#[cfg(feature = "jp2k")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BoxRef {
    kind: [u8; 4],
    /// First byte of the box, which is the first byte of its `LBox`.
    at: usize,
    /// First byte of the payload.
    start: usize,
    /// One past the last byte of the payload, and of the box.
    end: usize,
}

/// One box: a 32-bit `LBox` counting itself, the four-byte `TBox`, then the
/// payload.
///
/// The short form only, which is all [`wrap_bare_codestream`] needs: every box
/// it writes is a few dozen bytes and the codestream it wraps is already
/// bounded by the caller's allocation budget.
#[cfg(feature = "jp2k")]
fn jp2_box(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 8);
    let length = u32::try_from(payload.len() + 8).unwrap_or(u32::MAX);
    out.extend_from_slice(&length.to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(payload);
    out
}

/// The smallest JP2 container that carries `codestream` and says nothing else.
///
/// ISO/IEC 15444-1 Annex I: the signature box, a file-type box, the JP2 header
/// box, and the contiguous codestream. The header box is **empty**, which is
/// the point: every box that could name a colour space is one of its children,
/// so a header with no children is a file with no colour specification, which
/// is the shape [`ContainerLayout::unspecified_rewrite`] needs.
///
/// This is not a container to write out. It exists to give
/// `hayro-jpeg2000`'s JP2 route the same codestream its bare-codestream route
/// refuses, because that route synthesises an sRGB `colr` for anything with
/// three or more components and then validates the count against it.
#[cfg(feature = "jp2k")]
fn wrap_bare_codestream(codestream: &[u8]) -> Vec<u8> {
    let mut file_type = b"jp2 ".to_vec();
    file_type.extend_from_slice(&0u32.to_be_bytes());
    file_type.extend_from_slice(b"jp2 ");

    let mut out = JP2_SIGNATURE.to_vec();
    out.extend_from_slice(&jp2_box(b"ftyp", &file_type));
    out.extend_from_slice(&jp2_box(b"jp2h", &[]));
    out.extend_from_slice(&jp2_box(b"jp2c", codestream));
    out
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
            at,
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
            let mut enum_cs = None;
            let mut colr_placement = None;
            for header in top.iter().filter(|b| &b.kind == b"jp2h") {
                let boxes = walk_boxes(bytes, header.start, header.end)?;
                let colr: Vec<&[u8]> = boxes
                    .iter()
                    .filter(|b| &b.kind == b"colr")
                    .map(|b| &bytes[b.start..b.end])
                    .collect();
                icc = colr.iter().copied().find_map(icc_payload);
                enum_cs = boxes.iter().filter(|b| &b.kind == b"colr").find_map(|b| {
                    enumerated_colour_space(&bytes[b.start..b.end]).map(|cs| (cs, b.start + 3))
                });
                // The removal splices out a whole box and shrinks the `LBox`
                // above it, so it needs both ranges and a `jp2h` whose length
                // really is that `LBox`. A `jp2h` written with an extended or
                // to-the-end length is left alone rather than guessed at.
                if colr_placement.is_none() && header.start - header.at == 8 {
                    colr_placement =
                        boxes
                            .iter()
                            .find(|b| &b.kind == b"colr")
                            .map(|b| ColrPlacement {
                                at: b.at,
                                end: b.end,
                                header_at: header.at,
                                header_end: header.end,
                            });
                }
                if icc.is_some() || enum_cs.is_some() {
                    break;
                }
            }
            Ok(Self {
                codestream,
                icc,
                enum_cs_at: enum_cs.map(|(_, at)| at),
                enum_cs: enum_cs.map(|(cs, _)| cs),
                bare: false,
                colr: colr_placement,
            })
        } else if bytes.starts_with(CODESTREAM_SIGNATURE) {
            Ok(Self {
                codestream: 0,
                icc: None,
                enum_cs_at: None,
                enum_cs: None,
                bare: true,
                colr: None,
            })
        } else {
            Err(container(
                "the leading bytes are neither the JP2 signature box nor a SOC + SIZ pair",
            ))
        }
    }
}

#[cfg(feature = "jp2k")]
impl ContainerLayout {
    /// A copy of the file whose `METH = 1` `colr` box names a colour space the
    /// decoder handles without deciding anything, or `None` when the file
    /// already does.
    ///
    /// `hayro-jpeg2000` resolves the box itself, and on three of the values it
    /// meets that is a refusal or a conversion where `jp2kload` reads the file
    /// and leaves the samples alone (#771, #848, #849). Every decision the box
    /// makes belongs to this module now, the interpretation since #767 and the
    /// inverse YCC since #771, so the box is rewritten on the way in to say
    /// only what the component count already says.
    ///
    /// Only the boxes the decoder gets wrong are rewritten, and
    /// [`DECODER_SAFE_ENUMS`] is the closed set it gets right, each member
    /// pinned by a committed fixture. Narrowing it that far is the point:
    /// every file that decodes today hands the decoder byte-identical bytes
    /// and comes out with the digest it already has, and only files that used
    /// to be refused change route at all.
    ///
    /// sRGB is the neutral value for every component count, one included, and
    /// that is measured rather than assumed: `hayro-jpeg2000` reconciles an
    /// sRGB box against a one-component file by reporting `Gray` on its own,
    /// so the rewrite cannot invent a channel. Choosing by component count
    /// instead gets a palette wrong, where `SIZ` declares one component and
    /// `pclr` expands it to three.
    fn neutral_rewrite(&self, bytes: &[u8], header: &CodestreamHeader) -> Option<Vec<u8>> {
        let at = self.enum_cs_at?;
        if self.decoder_resolves_this_box(header) {
            return None;
        }
        let mut copy = bytes.to_vec();
        copy.get_mut(at..at + 4)?
            .copy_from_slice(&ENUMCS_SRGB.to_be_bytes());
        Some(copy)
    }

    /// A copy of the file that says nothing about its colour space, for the
    /// component counts the decoder cannot name, or `None` when there is no
    /// such copy to make.
    ///
    /// `hayro-jpeg2000` validates the component count against whatever colour
    /// space it resolved and gives up with `ValidationError::TooManyChannels`
    /// on anything it cannot map onto greyscale, RGB, CMYK or one of those
    /// plus alpha. That reads like a five-channel ceiling and is not one: with
    /// **no** colour specification box at all, `get_color_space` answers
    /// `ColorSpace::Unknown { num_channels }` for any count, and the
    /// validation that follows compares that count against itself and passes.
    /// So the rewrite is a removal rather than a substitution, and it is
    /// derived from the decoder's own fallback rather than invented until it
    /// stopped complaining.
    ///
    /// Two shapes, and a third that needs nothing:
    ///
    /// * a JP2 with a `colr` box loses it, and the `jp2h` above it loses the
    ///   same number of bytes from its `LBox`;
    /// * a bare codestream has no box to drop, so it is wrapped in the
    ///   smallest container the decoder's JP2 route accepts: the signature
    ///   box, a `jp2 ` file-type box, an **empty** `jp2h`, and the codestream.
    ///   `jp2h`'s children are the only boxes that carry colour, so an empty
    ///   one is the same statement as a missing `colr`;
    /// * a JP2 that already has no `colr` box never reaches here, because it
    ///   was never refused.
    ///
    /// The caller runs this **only** after the decoder has refused the file on
    /// the channel count, so no file that decodes today is handed different
    /// bytes, and a file this does not rescue keeps the error it already had.
    /// Measured against the oracle on 5, 6 and 8 components: `vips jp2kload`
    /// reads all three as `N bands, srgb` with every sample identical, which
    /// is what [`interpretation`] answers for
    /// `ColorSpace::Unknown { num_channels }` at three bands or more.
    fn unspecified_rewrite(&self, bytes: &[u8]) -> Option<Vec<u8>> {
        if self.bare {
            return Some(wrap_bare_codestream(bytes));
        }
        let colr = self.colr?;
        let width = colr.end.checked_sub(colr.at)?;
        let shrunk = u32::try_from(colr.header_end.checked_sub(colr.header_at)? - width).ok()?;
        let mut copy = Vec::with_capacity(bytes.len() - width);
        copy.extend_from_slice(bytes.get(..colr.at)?);
        copy.extend_from_slice(bytes.get(colr.end..)?);
        copy.get_mut(colr.header_at..colr.header_at + 4)?
            .copy_from_slice(&shrunk.to_be_bytes());
        Some(copy)
    }

    /// Whether the decoder resolves this `colr` box the way `jp2kload` does,
    /// so it can go through untouched.
    ///
    /// [`DECODER_SAFE_ENUMS`] is the set, with one condition on it: sYCC is
    /// safe only over **three** components, because that is what its transform
    /// reads. On any other count `hayro-jpeg2000` refuses with "failed to
    /// convert from sYCC to RGB", which is the same shape as the e-YCC refusal
    /// #848 filed, and vips's own answer there is the broken one, a three-band
    /// header whose pixels never arrive.
    ///
    /// A file with no enum at all, a bare codestream or a `METH = 2` profile
    /// box, has nothing to rewrite and nothing to get wrong, so it is safe by
    /// construction.
    fn decoder_resolves_this_box(&self, header: &CodestreamHeader) -> bool {
        match self.enum_cs {
            Some(ENUMCS_SYCC) => header.components.len() == 3,
            Some(cs) => DECODER_SAFE_ENUMS.contains(&cs),
            None => true,
        }
    }

    /// Whether this module runs the inverse YCC itself, given the codestream's
    /// `SIZ`.
    ///
    /// openjpeg's rule is that the transform runs when the resolved colour
    /// space is sYCC or e-YCC, and "resolved" means the `colr` box's enum
    /// where there is a recognised one and `opj_j2k_read_siz`'s heuristic
    /// (three components with the chroma pair subsampled) where there is not.
    /// Measured on 8.18.6 over `chroma_sub_on.jp2`, whose components are
    /// subsampled, by rewriting only its `colr` box:
    ///
    /// | box | `vips getpoint 0 0` | transform |
    /// |---|---|---|
    /// | `METH = 1`, sYCC (the file as committed) | `4 1 241` | yes |
    /// | `METH = 1`, sRGB | `29 248 110` | **no** |
    /// | `METH = 2`, a profile | `4 1 241` | **yes** |
    ///
    /// So a recognised non-YCC enum suppresses the subsampling heuristic, and
    /// a profile box does not, because it leaves the colour space exactly
    /// where `SIZ` put it. Before #771 this module asked `bare && subsampled`,
    /// which got the third row wrong.
    ///
    /// sYCC is the one case the decoder still handles, since its box is not
    /// rewritten, so this answers `false` there and lets it. e-YCC is
    /// rewritten, so the transform becomes this module's job.
    fn runs_inverse_ycc(&self, header: &CodestreamHeader) -> bool {
        // `sycc_to_rgb` reads three planes. vips reaches the same guard from
        // the other side: on a one-component file tagged sYCC it produces a
        // three-band header the pixels never fill, and any read of one fails.
        if header.components.len() != 3 {
            return false;
        }
        match self.enum_cs {
            // Rewritten to sRGB on the way in, so nothing else will do it.
            Some(ENUMCS_EYCC) => true,
            // Left alone at three components, so the decoder does it, as it
            // always has. Any other count returned above.
            Some(ENUMCS_SYCC) => false,
            // A recognised space that is not YCC suppresses the heuristic.
            Some(cs) if DECODER_SAFE_ENUMS.contains(&cs) => false,
            // Unspecified: a bare codestream, a profile box, or an enum nobody
            // recognises. `SIZ` decides, which is openjpeg's rule.
            _ => header.chroma_subsampled(),
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

/// The `EnumCS` inside a `METH=1` `colr` box, or `None` for any other method.
///
/// The other half of [`icc_payload`], reading the same three-byte prelude the
/// other way: `METH = 1` means the next four bytes are an enumerated colour
/// space rather than a profile. A box too short to hold one is `None` rather
/// than an error, the way a `METH=2` box too short to hold a profile is, since
/// the decoder is the thing that decides whether the file is readable at all.
#[cfg(feature = "jp2k")]
fn enumerated_colour_space(payload: &[u8]) -> Option<u32> {
    match payload.first() {
        Some(1) if payload.len() >= 7 => Some(u32::from_be_bytes([
            payload[3], payload[4], payload[5], payload[6],
        ])),
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
    /// [`ContainerLayout::runs_inverse_ycc`]'s job and not this one's.
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

    // The carrier decisions, answered off the codestream's own SIZ rather
    // than off the decoder, because `hayro-jpeg2000` reports neither the sign
    // bit nor a per-component precision before the decode, and both decide
    // what the file can be carried as.
    //
    // The sign bit is a whole-image property here and a per-component one in
    // the format, so a file whose components disagree has no carrier at all.
    // vips refuses that file too, which is why it is a typed refusal rather
    // than a widening.
    let signed = header.components[0].signed;
    if let Some((signed_at, unsigned_at)) = header
        .components
        .iter()
        .position(|c| c.signed)
        .zip(header.components.iter().position(|c| !c.signed))
    {
        return Err(Jp2kError::MixedComponentSignedness {
            signed: signed_at,
            unsigned: unsigned_at,
        }
        .into());
    }
    for (component, spec) in header.components.iter().enumerate() {
        if spec.precision > MAX_PRECISION {
            return Err(Jp2kError::PrecisionNotSupported {
                component,
                precision: spec.precision,
                max: MAX_PRECISION,
            }
            .into());
        }
    }
    // The one combination where a signed file has an oracle answer and the
    // answer is unusable. `sycc_to_rgb` below is written against the
    // unsigned domain, and vips runs it against the signed one anyway, in
    // the component's own carrier: the offset subtraction wraps and the
    // clamp is to the unsigned range. Refused rather than reproduced, with
    // the numbers on the variant.
    if signed && layout.runs_inverse_ycc(&header) {
        return Err(Jp2kError::SignedInverseYcc {
            components: header.components.len(),
        }
        .into());
    }

    // What the decoder is allowed to see. `hayro-jpeg2000` resolves the `colr`
    // box itself and refuses or converts on what it cannot map: an unrecognised
    // enum (#771), e-YCC (#848) and CIELab (#849) all stop a decode that the
    // codestream is perfectly capable of. None of that is the codestream's
    // doing, and none of it is what `jp2kload` does, so the enum is replaced
    // with a neutral one before the file goes in. This module already owns
    // every decision the box makes: the interpretation since #767, and the
    // inverse YCC below.
    //
    // Neutralised rather than stripped, because the boxes around it carry
    // things the decode genuinely needs: `pclr` and `cmap` for a palette,
    // `cdef` for alpha. Handing over the bare codestream would lose all three.
    let neutralised;
    let for_decoder: &[u8] = match layout.neutral_rewrite(bytes, &header) {
        Some(copy) => {
            // The one allocation this function makes that is not the raster,
            // so it goes through the same budget. It is a copy of an input
            // already resident, and JPEG 2000 files are small against the
            // rasters they decode to, but "small" is not a bound and the
            // caller set one.
            limits.check_alloc("JPEG 2000 container rewrite", copy.len() as u64)?;
            neutralised = copy;
            &neutralised
        }
        None => bytes,
    };
    // The second rewrite, and the only one that changes which files decode at
    // all. `hayro-jpeg2000` refuses a component count it cannot map onto a
    // colour space it can name, which is every count above four, and the way
    // past it is to stop naming one: with no colour specification the decoder
    // reaches its own `Unknown { num_channels }` arm and validates the count
    // against itself. Issue #769, and the sweep is on
    // `ContainerLayout::unspecified_rewrite`.
    //
    // Gated on the refusal rather than on the count, which is what makes it
    // safe: a five-component CMYK-plus-alpha file is a count the decoder
    // *can* name, and dropping its box would turn a file that decodes into
    // one that does not. Asking first costs one header parse and only for
    // files that were about to be refused anyway.
    let unspecified;
    let image = match Jp2kImage::new(for_decoder, &DecodeSettings::default()) {
        Ok(image) => image,
        Err(refusal) => {
            let retry = matches!(
                refusal,
                hayro_jpeg2000::DecodeError::Validation(
                    hayro_jpeg2000::ValidationError::TooManyChannels
                )
            )
            .then(|| layout.unspecified_rewrite(for_decoder))
            .flatten();
            let Some(copy) = retry else {
                return Err(decode_error(refusal).into());
            };
            limits.check_alloc("JPEG 2000 container rewrite", copy.len() as u64)?;
            unspecified = copy;
            // A file the removal does not rescue keeps the error it already
            // had, so nothing here can report a second-order failure about a
            // container this module synthesised.
            Jp2kImage::new(&unspecified, &DecodeSettings::default())
                .map_err(|_| decode_error(refusal))?
        }
    };
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
    // Keyed on the sample kind rather than on the byte width, because a
    // width cannot tell `Int8` from `Gray8` or `Int16` from `Gray16`, and
    // the sign bit is exactly the thing that decides which one this is
    // (issues #607, #905).
    let kind = match (element_bytes, signed) {
        (1, false) => SampleKind::U8,
        (1, true) => SampleKind::I8,
        (_, false) => SampleKind::U16,
        (_, true) => SampleKind::I16,
    };
    let format = carrier(bands, kind)?;

    // The allocation budget, which `check_pixels` does not imply: a
    // 1-gigapixel `max_pixels` permits an 8 GiB `Rgba16` frame against a
    // 512 MiB default budget. Priced from the declared geometry before the
    // decoder reserves anything, the way `crate::exr` and `crate::fits` price
    // theirs, and reported through the one shared shape (issue #686).
    //
    // Plus what `hayro-jpeg2000` holds beside the buffer this module fills,
    // which is much the larger half and is why this was the worst of the
    // three decoders #944 measured: a 1024x1024 RGB file peaked at **6.45x**
    // its price, and a 4096x4096 one at 244 MiB against a 48 MiB budget.
    // `max_alloc_bytes` is documented as a ceiling on peak memory, so a
    // caller sizing a container limit from it was killed by that factor, and
    // the refusal message understated by the same one.
    //
    // The decoder has no budget of its own to hand this one to.
    // `DecodeSettings` carries no allocation limit at all, which is why this
    // takes `crate::webp`'s route rather than `crate::jxl`'s: jxl wires the
    // caller's ceiling into `jxl-oxide`'s `AllocTracker` and lets the
    // dependency refuse itself, and there is nothing here to wire it into.
    //
    // Two terms, both measured with a counting global allocator in
    // `tests/decode_working_set.rs`:
    //
    // * **Per image**, one `f32` sample per component for the decoded
    //   component data, and a second copy of all of them while the `cdef`
    //   box's channel reorder clones the set. Eight bytes per band-pixel.
    // * **Per tile**, the coefficient storage the code-block pass works in,
    //   which `build` reallocates per tile and which therefore scales with
    //   the *tile* rather than with the image, plus its bookkeeping. Measured
    //   at 5.8 bytes per band-tile-pixel across four geometries and two
    //   sample depths, priced at ten.
    //
    // The tile term is why a 512x512 file costs proportionally more than a
    // 4096x4096 one: `jp2ksave` tiles on a 512 grid, so past that size the
    // per-tile buffers stop growing. `min` with the image is the upper bound
    // for the first tile, whose rectangle a non-zero `XTOsiz` can shrink but
    // never grow.
    let plane = u64::from(width).saturating_mul(u64::from(height));
    let tile_plane = u64::from(header.tile_width.clamp(1, width.max(1)))
        .saturating_mul(u64::from(header.tile_height.clamp(1, height.max(1))));
    let working_set = u64::from(bands).saturating_mul(
        plane
            .saturating_mul(8)
            .saturating_add(tile_plane.saturating_mul(10)),
    );
    limits.check_image_alloc_with_working_set(
        "JPEG 2000 component buffers",
        width,
        height,
        u64::from(bands),
        element_bytes,
        working_set,
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
    let ycc = layout.runs_inverse_ycc(&header);
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
            // `hayro-jpeg2000` applies the DC level shift to every component
            // whether or not `SIZ` set the sign bit, so a signed component
            // arrives in the unsigned range and the file's own sample is
            // that value minus half the range. Undoing it here rather than
            // in `quantise` keeps one clamp doing both jobs: clamping to
            // `[0, 2^p - 1]` and then subtracting `2^(p-1)` is exactly a
            // clamp to the signed range `[-2^(p-1), 2^(p-1) - 1]`.
            let dc_shift = if signed { 1i32 << (precision - 1) } else { 0 };
            for (i, sample) in data.samples().iter().enumerate() {
                // The left-justification is the same shift for both signs:
                // it multiplies by a power of two, and two's complement
                // multiplication is sign-agnostic. `-2048 << 4` is -32768,
                // which is what vips reports for `depth12s.j2k`.
                let value = ((quantise(*sample, precision) as i32 - dc_shift) << shift) as u32;
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
    // The `colr` box's enumerated colour space decides this where there is
    // one and openjpeg recognises it, because that is what `jp2kload` reads
    // and the component count is not (#767). Everything else, which is a bare
    // codestream, a `METH=2` profile box, or an enum openjpeg does not know,
    // falls back to the decoder's resolved colour space and the band-count
    // guess inside it, which is what vips's UNSPECIFIED arm does too.
    raster.meta.interpretation = Some(
        layout
            .enum_cs
            .and_then(|enumcs| enumerated_interpretation(enumcs, element_bytes))
            .unwrap_or_else(|| interpretation(image.color_space(), element_bytes)),
    );
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

/// The sample carrier for a band count and a sample kind.
///
/// Keyed on the kind rather than on a byte width, because the width is not
/// enough: one byte is `Gray8` or `Int8` and two is `Gray16` or `Int16`, and
/// only the codestream's sign bit says which. [`PixelFormat::with_kind`] is
/// the constructor that cannot be asked that question ambiguously, and it
/// keeps the unsigned mapping unchanged, so 1, 3 and 4 bands still reach the
/// named `Gray` / `Rgb` / `Rgba` variants and everything else reaches the
/// multiband ones.
///
/// A zero band count is the one thing that has no carrier, and it is
/// unreachable from a valid codestream because `SIZ` refuses `Csiz = 0`
/// before this is called; the check is here anyway because the count comes
/// from the decoder rather than from `SIZ`, and a palette resolves to
/// whatever the palette says.
#[cfg(feature = "jp2k")]
fn carrier(bands: u32, kind: SampleKind) -> Result<PixelFormat, Jp2kError> {
    let max = u32::from(u16::MAX);
    usize::try_from(bands)
        .ok()
        .and_then(|n| PixelFormat::with_kind(n, kind))
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

/// `EnumCS` for CMYK, the one enumerated space whose tag the element width
/// does not change.
#[cfg(feature = "jp2k")]
const ENUMCS_CMYK: u32 = 12;
/// `EnumCS` for sRGB, and the neutral value handed to the decoder for a file
/// with three or more components.
#[cfg(feature = "jp2k")]
const ENUMCS_SRGB: u32 = 16;
/// `EnumCS` for greyscale, and the neutral value for one or two components.
#[cfg(feature = "jp2k")]
const ENUMCS_GREY: u32 = 17;
/// `EnumCS` for sYCC, which turns the inverse YCC on.
#[cfg(feature = "jp2k")]
const ENUMCS_SYCC: u32 = 18;
/// `EnumCS` for e-YCC, which vips answers exactly as it answers sYCC.
#[cfg(feature = "jp2k")]
const ENUMCS_EYCC: u32 = 24;

/// The enumerated colour spaces `hayro-jpeg2000` resolves the way `jp2kload`
/// does, so their `colr` box reaches the decoder untouched.
///
/// A closed set on purpose, and every member is pinned by a committed
/// fixture: CMYK by `cmyk_lossless.jp2`, sRGB by `rgb_lossless.jp2`,
/// `rgba_lossless.jp2` and `chroma_sub_off.jp2`, greyscale by
/// `grey_tile8.jp2`, sYCC by `chroma_sub_on.jp2` and
/// `chroma_tiny_sub_on.jp2`. Everything outside it is rewritten by
/// [`ContainerLayout::neutral_rewrite`], which is what keeps #771, #848 and
/// #849 from being a change to any file that already decoded.
#[cfg(feature = "jp2k")]
const DECODER_SAFE_ENUMS: [u32; 4] = [ENUMCS_CMYK, ENUMCS_SRGB, ENUMCS_GREY, ENUMCS_SYCC];

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
fn enumerated_interpretation(enumcs: u32, element_bytes: u64) -> Option<Interpretation> {
    let wide = element_bytes > 1;
    Some(match enumcs {
        // CMYK is the one answer the element width does not change: vips
        // reports `cmyk` for the 8-bit and the 16-bit file alike.
        ENUMCS_CMYK => Interpretation::Cmyk,
        // Greyscale. Three components tagged with it come back `b-w`, which is
        // the row that proves the band count is not consulted.
        ENUMCS_GREY if wide => Interpretation::Grey16,
        ENUMCS_GREY => Interpretation::Bw,
        // sRGB, sYCC and e-YCC all land on the RGB tag. The last two also turn
        // the inverse YCC on, which `decode` decides, not this function.
        ENUMCS_SRGB | ENUMCS_SYCC | ENUMCS_EYCC if wide => Interpretation::Rgb16,
        ENUMCS_SRGB | ENUMCS_SYCC | ENUMCS_EYCC => Interpretation::Srgb,
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
///
/// One codestream per tile, spliced under one main header. JPEG 2000 codes
/// every tile independently, so the tile-part for a region is the same bytes
/// whether it was produced as one tile of a tiled image or as the whole of a
/// standalone one placed at the same absolute grid coordinates, which is what
/// makes the splice a rearrangement rather than a re-encode. The
/// [`SaveOptions`] doc has the byte-identity measurement against
/// `vips jp2ksave` that says this is the same thing OpenJPEG does.
#[cfg(feature = "jp2k")]
fn encode(raster: &Raster, options: SaveOptions) -> Result<Vec<u8>, EncodeError> {
    use openjpeg2_pure::{EncodeOptions, Encoder, Format, Image, ImageComponent};

    let SaveOptions {
        compression,
        tile_width,
        tile_height,
    } = options;
    let ComponentLayout {
        precision,
        element_bytes,
        signed,
    } = sample_depth(raster.format())?;
    let (width, height) = (raster.width(), raster.height());
    let bands = raster.format().channels();
    // The ceiling is the format's, not the loader's. It was the loader's
    // until #769: `hayro-jpeg2000` refuses any component count it cannot map
    // onto a colour space it can name, so anything past four was a file this
    // crate could write and could not read back. `decode` now hands that file
    // to the decoder with nothing saying what its colour space is, which is
    // the arm the decoder answers `Unknown { num_channels }` from, so both
    // halves move together and the encoder stops at `Csiz`.
    if bands > MAX_BANDS {
        return Err(EncodeError::encode(format!(
            "jp2k: a JPEG 2000 codestream declares at most {MAX_BANDS} components in \
             its SIZ marker (Csiz, ISO/IEC 15444-1 Table A.9) and the raster has \
             {bands}; measured, openjpeg2-pure-rs refuses {} with EncoderSetupFailed",
            MAX_BANDS + 1
        )));
    }

    // The tile grid, which is `jp2ksave`'s and defaults to its 512x512 rather
    // than to the whole image: measured, vips writes `tile-width: 512` for a
    // 600x600 image, so an encoder that always writes one tile diverges on
    // every image bigger than that.
    let across = width.div_ceil(tile_width.get()).max(1);
    let down = height.div_ceil(tile_height.get()).max(1);
    let tiles = u64::from(across) * u64::from(down);
    if tiles > MAX_TILES {
        return Err(EncodeError::encode(format!(
            "jp2k: a {width}x{height} image at {}x{} pixels per tile is {tiles} tiles \
             and a codestream indexes at most {MAX_TILES} through its two-byte Isot; \
             vips jp2ksave refuses the same grid with \"Invalid number of tiles\"",
            tile_width.get(),
            tile_height.get()
        )));
    }

    let (irreversible, rates) = match compression {
        // `tcp_rates[0] = 0` under `cp_disto_alloc` is OpenJPEG's spelling for
        // "no rate target", which with the reversible 5/3 wavelet is a
        // lossless encode.
        Compression::Lossless => (false, vec![0.0f32]),
        Compression::Lossy { ratio } => (true, vec![f32::from(ratio.get())]),
    };
    let colour = encoder_colour_space(raster);
    let data = raster.data();
    let row = width as usize * bands * element_bytes;

    // One tile, encoded as a standalone image sitting where the tile sits.
    //
    // `x0` / `y0` are the whole of what makes the splice legal. OpenJPEG's
    // tile grid starts at `cp_tx0`, which is zero, so a standalone image whose
    // own origin is the tile's origin is one tile covering exactly the
    // absolute region that tile covers. The wavelet, the code-block partition
    // and the precincts are all anchored on those absolute coordinates, so the
    // coded bytes are the ones a tiled encode would have produced.
    let encode_tile = |tx: u32, ty: u32, format: Format| -> Result<Vec<u8>, EncodeError> {
        let x0 = tx * tile_width.get();
        let y0 = ty * tile_height.get();
        let x1 = x0.saturating_add(tile_width.get()).min(width);
        let y1 = y0.saturating_add(tile_height.get()).min(height);
        let (tile_w, tile_h) = (x1 - x0, y1 - y0);

        // One plane per band, de-interleaved, because JPEG 2000 is a
        // component-planar format and the encoder takes it that way.
        let plane = tile_w as usize * tile_h as usize;
        let mut components = Vec::with_capacity(bands);
        for band in 0..bands {
            let mut samples = Vec::with_capacity(plane);
            for line in 0..tile_h as usize {
                let base = (y0 as usize + line) * row + x0 as usize * bands * element_bytes;
                for column in 0..tile_w as usize {
                    let at = base + (column * bands + band) * element_bytes;
                    // Read at the raster's own signedness, not at its width.
                    // Widening an `Int8` sample of -5 through `u8` would hand
                    // the encoder 251 and write a file that says -5 nowhere,
                    // which is the silent wrong answer issue #905 exists to
                    // avoid.
                    samples.push(match (element_bytes, signed) {
                        (1, false) => i32::from(data[at]),
                        (1, true) => i32::from(data[at] as i8),
                        (_, false) => i32::from(u16::from_ne_bytes([data[at], data[at + 1]])),
                        (_, true) => i32::from(i16::from_ne_bytes([data[at], data[at + 1]])),
                    });
                }
            }
            components.push(
                ImageComponent::new(tile_w, tile_h, precision, signed, samples)
                    .map_err(|e| EncodeError::encode(format!("jp2k: component {band}: {e:?}")))?,
            );
        }

        let mut image = Image::new(tile_w, tile_h, colour, components)
            .map_err(|e| EncodeError::encode(format!("jp2k: {e:?}")))?;
        image.x0 = x0;
        image.y0 = y0;
        Encoder::encode(
            &image,
            &EncodeOptions {
                // `jp2ksave` hard-codes `OPJ_CODEC_JP2` and writes the same
                // container for all five suffixes it registers, so there is
                // nothing here for a caller to choose. The tiles after the
                // first are asked for the bare codestream instead, because
                // the container is written once: measured, a `Jp2` encode's
                // `jp2c` payload is byte-identical to the `J2k` encode of the
                // same image.
                format,
                // libviprs schedules its own work in the engine, so a codec
                // that starts a second pool underneath it is not something
                // this crate wants. Same call as `exr`'s and `jxl-oxide`'s
                // dropped `rayon`.
                threads: 0,
                irreversible,
                // `jp2ksave` sets `tcp_mct` from the band count alone
                // (`image->Bands >= 3`), CMYK included: measured, `mct` is 1
                // in the `cmyk_lossless.jp2` and `rgba_lossless.jp2`
                // fixtures' `COD` segments. The reversible multiple-component
                // transform is exactly invertible, so this costs no accuracy:
                // measured, every carrier round-trips through
                // `vips jp2kload` bit for bit with it on.
                use_mct: bands >= 3,
                rates: rates.clone(),
                // From the **image**, not from the tile, which is `jp2ksave`'s
                // own rule. It is also what makes one main header serve every
                // tile, and it is why a tile narrower than the decomposition
                // needs is refused by the encoder rather than quietly given a
                // different one: vips fails that grid too, with "Number of
                // resolutions is too high in comparison to the size of tiles".
                num_resolutions: Some(num_resolutions(width, height)),
            },
        )
        .map_err(|e| {
            if tiles == 1 {
                EncodeError::encode(format!("jp2k: {e:?}"))
            } else {
                EncodeError::encode(format!("jp2k: tile ({tx}, {ty}): {e:?}"))
            }
        })
    };

    let mut file = encode_tile(0, 0, Format::Jp2)?;
    let own = OwnContainer::parse(&file)?;
    // vips writes the requested tile size into `SIZ` whether or not it cuts
    // the image in two: measured, `jp2ksave --lossless` on a 37x21 image
    // writes `XTsiz = YTsiz = 512`, and this module used to write 37 and 21.
    // That is the only byte in a single-tile file that differed from vips's.
    patch_siz(
        &mut file,
        own.codestream,
        width,
        height,
        tile_width.get(),
        tile_height.get(),
    )?;
    if tiles == 1 {
        return splice_channel_definition(file, raster.interpretation(), bands);
    }

    // More than one tile, so the container and the main header the first tile
    // came with describe a 37x21-shaped lie about a 16x16 encode. Both say the
    // image size and nothing else about the geometry, so both are corrected in
    // place rather than rebuilt.
    patch_ihdr(&mut file, own.ihdr, width, height)?;
    let (first_sot, first_end) = sole_tile_part(&file[own.codestream..])?;
    let main_header = own.codestream..own.codestream + first_sot;
    let after_siz = own.codestream + siz_length(&file[own.codestream..])?;
    let shared: Vec<u8> = file[after_siz..main_header.end].to_vec();

    let mut out = file[..main_header.end].to_vec();
    append_tile_part(
        &mut out,
        &file[own.codestream + first_sot..own.codestream + first_end],
        0,
    )?;
    for index in 1..tiles {
        let tx = u32::try_from(index % u64::from(across)).expect("a tile index fits");
        let ty = u32::try_from(index / u64::from(across)).expect("a tile index fits");
        let part = encode_tile(tx, ty, Format::J2k)?;
        // The main header is written once and claimed to serve every tile, so
        // every tile has to agree with it. They do by construction (the
        // coding style comes from the image, not from the tile) and this is
        // what says so rather than assuming it.
        let (sot, end) = sole_tile_part(&part)?;
        if part[siz_length(&part)?..sot] != shared[..] {
            return Err(EncodeError::encode(format!(
                "jp2k: tile ({tx}, {ty}) encoded with a different coding style than tile \
                 (0, 0), so one main header cannot describe both"
            )));
        }
        append_tile_part(&mut out, &part[sot..end], index)?;
    }
    out.extend_from_slice(&MARKER_EOC.to_be_bytes());

    // `jp2c` is the last box, so its length is everything from its own first
    // byte to the end.
    let jp2c_length = u32::try_from(out.len() - own.jp2c_at).map_err(|_| {
        EncodeError::encode("jp2k: the tiled codestream is too long for a 32-bit box length")
    })?;
    out[own.jp2c_at..own.jp2c_at + 4].copy_from_slice(&jp2c_length.to_be_bytes());
    // Last, because it moves `jp2c` along without changing its length: the box
    // counts only itself.
    splice_channel_definition(out, raster.interpretation(), bands)
}

/// The `cdef` box `jp2ksave` writes for this interpretation and band count, or
/// `None` for the shapes it writes none for.
///
/// Measured over six band counts and six interpretations on
/// `/opt/homebrew/bin/vips` 8.18.6, reading the `jp2h` children back out of
/// each file. Exactly two of the 36 shapes get a box:
///
/// | bands | `b-w` / `grey16` | `srgb` / `rgb16` | `cmyk` | `multiband` |
/// |---|---|---|---|---|
/// | 1 | none | none | none | none |
/// | 2 | **`(0,0,1) (1,1,0)`** | none | none | none |
/// | 3 | none | none | none | none |
/// | 4 | none | **`(0,0,1) (1,0,2) (2,0,3) (3,1,0)`** | none | none |
/// | 5 | none | none | none | none |
/// | 6 | none | none | none | none |
///
/// So it is **not** `vips_image_hasalpha`, which is true for two bands
/// whatever the interpretation and for anything past four. It is greyscale
/// plus one and RGB plus one and nothing else: CMYK plus one gets no box, and
/// neither does a five- or six-band image under any tag.
///
/// Keyed on the interpretation rather than on the band count, which is the
/// same question [`encoder_colour_space`] asks and for the same reason: a
/// four-band CMYK raster and a four-band RGBA one are the same
/// [`PixelFormat`] and want different answers. Keying it on the enumerated
/// colour space instead would get the multiband column wrong, because this
/// module writes greyscale for a two-band raster where vips writes
/// unspecified, so a `Multi8(2)` would pick up an alpha channel it never
/// claimed to have.
///
/// The entries are `Cn`, `Typ`, `Asoc` (I.5.3.6): each colour channel is
/// `Typ = 0` with a one-based association, and the alpha channel is
/// `Typ = 1, Asoc = 0`, which associates it with the whole image.
#[cfg(feature = "jp2k")]
fn channel_definition(interpretation: Interpretation, bands: usize) -> Option<Vec<u8>> {
    let colours = match interpretation {
        Interpretation::Bw | Interpretation::Grey16 => 1usize,
        Interpretation::Srgb | Interpretation::Rgb16 => 3,
        _ => return None,
    };
    if bands != colours + 1 {
        return None;
    }
    let count = u16::try_from(bands).ok()?;
    let mut payload = count.to_be_bytes().to_vec();
    for channel in 0..count {
        let (kind, association) = if usize::from(channel) < colours {
            (0u16, channel + 1)
        } else {
            (1, 0)
        };
        payload.extend_from_slice(&channel.to_be_bytes());
        payload.extend_from_slice(&kind.to_be_bytes());
        payload.extend_from_slice(&association.to_be_bytes());
    }
    Some(jp2_box(b"cdef", &payload))
}

/// Add the `cdef` box to a JP2 this module just wrote, if this shape gets one.
///
/// It goes last inside `jp2h`, which is after `colr`, because that is where
/// `jp2ksave` puts it. Everything below `jp2h` moves along by the box's
/// length and nothing needs rewriting for it: `jp2c`'s own `LBox` counts only
/// itself, and no offset inside the codestream is a file offset.
#[cfg(feature = "jp2k")]
fn splice_channel_definition(
    file: Vec<u8>,
    interpretation: Interpretation,
    bands: usize,
) -> Result<Vec<u8>, EncodeError> {
    let Some(cdef) = channel_definition(interpretation, bands) else {
        return Ok(file);
    };
    let boxes = walk_boxes(&file, 0, file.len()).map_err(|e| {
        EncodeError::encode(format!(
            "jp2k: the container this module wrote does not parse: {e}"
        ))
    })?;
    let header = boxes
        .iter()
        .find(|b| &b.kind == b"jp2h")
        .ok_or_else(|| EncodeError::encode("jp2k: the container has no jp2h box"))?;
    if header.start - header.at != 8 {
        return Err(EncodeError::encode(
            "jp2k: the jp2h box this module wrote does not carry a plain length",
        ));
    }
    let grown = u32::try_from(header.end - header.at + cdef.len())
        .map_err(|_| EncodeError::encode("jp2k: the jp2h box does not fit a 32-bit length"))?;
    let mut out = Vec::with_capacity(file.len() + cdef.len());
    out.extend_from_slice(&file[..header.at]);
    out.extend_from_slice(&grown.to_be_bytes());
    out.extend_from_slice(&file[header.at + 4..header.end]);
    out.extend_from_slice(&cdef);
    out.extend_from_slice(&file[header.end..]);
    Ok(out)
}

/// The offsets [`encode`] rewrites in a JP2 it just wrote itself.
///
/// Every field is checked rather than assumed, because a container this module
/// then edits in place is one where a wrong offset is a corrupt file rather
/// than an error. The encoder is the only producer these bytes ever come from,
/// so anything here failing means it changed shape underneath this module.
#[cfg(feature = "jp2k")]
struct OwnContainer {
    /// First byte of the `jp2c` box, which is the first byte of the `LBox`
    /// the tiled assembly rewrites.
    jp2c_at: usize,
    /// First byte of the codestream, which is its `SOC` marker.
    codestream: usize,
    /// First byte of the `ihdr` box's payload, whose first eight bytes are
    /// the image height and width.
    ihdr: usize,
}

#[cfg(feature = "jp2k")]
impl OwnContainer {
    /// The `ihdr` payload's length, ISO/IEC 15444-1 Annex I.5.3.1: `HEIGHT`,
    /// `WIDTH`, `NC`, `BPC`, `C`, `UnkC`, `IPR`.
    const IHDR_PAYLOAD: usize = 14;

    fn parse(file: &[u8]) -> Result<Self, EncodeError> {
        let own = |reason: String| EncodeError::encode(format!("jp2k: {reason}"));
        let top = walk_boxes(file, 0, file.len()).map_err(|e| {
            own(format!(
                "the container this module wrote does not parse: {e}"
            ))
        })?;
        let jp2c = top
            .iter()
            .find(|b| &b.kind == b"jp2c")
            .ok_or_else(|| own("the container this module wrote has no jp2c box".into()))?;
        if jp2c.start - jp2c.at != 8 || jp2c.end != file.len() {
            return Err(own(
                "the jp2c box this module wrote is not the last box with a plain length".into(),
            ));
        }
        let header = top
            .iter()
            .find(|b| &b.kind == b"jp2h")
            .ok_or_else(|| own("the container this module wrote has no jp2h box".into()))?;
        let children = walk_boxes(file, header.start, header.end).map_err(|e| {
            own(format!(
                "the jp2h box this module wrote does not parse: {e}"
            ))
        })?;
        let ihdr = children
            .iter()
            .find(|b| &b.kind == b"ihdr")
            .ok_or_else(|| own("the jp2h box this module wrote has no ihdr box".into()))?;
        if ihdr.end - ihdr.start != Self::IHDR_PAYLOAD {
            return Err(own(format!(
                "the ihdr box this module wrote carries {} payload bytes and Annex I.5.3.1 \
                 fixes it at {}",
                ihdr.end - ihdr.start,
                Self::IHDR_PAYLOAD
            )));
        }
        Ok(Self {
            jp2c_at: jp2c.at,
            codestream: jp2c.start,
            ihdr: ihdr.start,
        })
    }
}

/// How long the `SOC` + `SIZ` pair opening a codestream is.
///
/// `SOC` is two bytes with no length and `SIZ` is a marker plus a length that
/// counts itself, so the first marker after them is at `2 + 2 + Lsiz`.
#[cfg(feature = "jp2k")]
fn siz_length(codestream: &[u8]) -> Result<usize, EncodeError> {
    let lsiz = codestream
        .get(4..6)
        .ok_or_else(|| EncodeError::encode("jp2k: the codestream ends inside its SIZ marker"))?;
    Ok(4 + usize::from(u16::from_be_bytes([lsiz[0], lsiz[1]])))
}

/// Rewrite the image and tile geometry in a codestream's `SIZ` marker.
///
/// The eight consecutive big-endian `u32`s after `Rsiz`, in the order Table
/// A.9 gives them: `Xsiz`, `Ysiz`, `XOsiz`, `YOsiz`, `XTsiz`, `YTsiz`,
/// `XTOsiz`, `YTOsiz`. The origins are all zero because this encoder writes
/// images at the grid origin and tile grids that start there.
#[cfg(feature = "jp2k")]
fn patch_siz(
    file: &mut [u8],
    codestream: usize,
    width: u32,
    height: u32,
    tile_width: u32,
    tile_height: u32,
) -> Result<(), EncodeError> {
    let opening = file
        .get(codestream..codestream + 4)
        .ok_or_else(|| EncodeError::encode("jp2k: the codestream ends before its SIZ marker"))?;
    if u16::from_be_bytes([opening[0], opening[1]]) != MARKER_SOC
        || u16::from_be_bytes([opening[2], opening[3]]) != MARKER_SIZ
    {
        return Err(EncodeError::encode(
            "jp2k: the codestream this module wrote does not open with SOC then SIZ",
        ));
    }
    // `SOC` (2), the `SIZ` marker (2), `Lsiz` (2), `Rsiz` (2).
    let at = codestream + 8;
    let fields = [width, height, 0, 0, tile_width, tile_height, 0, 0];
    let room = file
        .get_mut(at..at + fields.len() * 4)
        .ok_or_else(|| EncodeError::encode("jp2k: the SIZ marker ends before its geometry"))?;
    for (slot, value) in room.as_chunks_mut::<4>().0.iter_mut().zip(fields) {
        *slot = value.to_be_bytes();
    }
    Ok(())
}

/// Rewrite the image size in an `ihdr` box, which the first tile's encode
/// filled in with the first tile's size.
///
/// The current values are asserted rather than overwritten blind: they are the
/// tile-zero geometry this module just asked for, and anything else means the
/// box being patched is not the one this module thinks it is.
#[cfg(feature = "jp2k")]
fn patch_ihdr(file: &mut [u8], ihdr: usize, width: u32, height: u32) -> Result<(), EncodeError> {
    let payload = file
        .get_mut(ihdr..ihdr + 8)
        .ok_or_else(|| EncodeError::encode("jp2k: the ihdr box ends before its size fields"))?;
    payload[..4].copy_from_slice(&height.to_be_bytes());
    payload[4..].copy_from_slice(&width.to_be_bytes());
    Ok(())
}

/// The one tile-part of a single-tile codestream: where its `SOT` marker
/// starts, and where the part ends.
///
/// The walk is the main header's, stopping at the first `SOT`. `Psot` is the
/// part's whole length counted from the `SOT` marker; a `Psot` of zero means
/// "to the `EOC`", which this encoder does not write but the standard allows.
#[cfg(feature = "jp2k")]
fn sole_tile_part(codestream: &[u8]) -> Result<(usize, usize), EncodeError> {
    let bad = |reason: &str| EncodeError::encode(format!("jp2k: {reason}"));
    let mut at = 2usize;
    while at + 4 <= codestream.len() {
        let marker = u16::from_be_bytes([codestream[at], codestream[at + 1]]);
        if marker == MARKER_SOT {
            let psot = codestream
                .get(at + 6..at + 10)
                .ok_or_else(|| bad("a SOT segment ends before its Psot field"))?;
            let psot = u32::from_be_bytes([psot[0], psot[1], psot[2], psot[3]]) as usize;
            let end = if psot == 0 {
                codestream
                    .len()
                    .checked_sub(2)
                    .ok_or_else(|| bad("a codestream with Psot = 0 has no EOC to end at"))?
            } else {
                at.checked_add(psot)
                    .filter(|end| *end <= codestream.len())
                    .ok_or_else(|| bad("a tile-part declares a Psot past the end"))?
            };
            if end < at + 12 {
                return Err(bad("a tile-part is shorter than its own SOT segment"));
            }
            return Ok((at, end));
        }
        if marker == MARKER_SOD || marker == MARKER_EOC || marker >> 8 != 0xff {
            return Err(bad("a codestream this module wrote has no tile-part"));
        }
        let length = usize::from(u16::from_be_bytes([codestream[at + 2], codestream[at + 3]]));
        if length < 2 {
            return Err(bad(
                "a marker segment declares a length shorter than itself",
            ));
        }
        at += 2 + length;
    }
    Err(bad(
        "a codestream this module wrote ends inside its main header",
    ))
}

/// Append one tile-part, renumbered for its place in the assembled grid.
///
/// `Isot` becomes the tile's index, `Psot` the part's own length, and
/// `TPsot` / `TNsot` say this is the first and only tile-part for the tile.
/// Every one of them is already what it needs to be except `Isot`, which each
/// standalone encode wrote as zero; they are written anyway so the assembled
/// header says what it means rather than what it inherited.
#[cfg(feature = "jp2k")]
fn append_tile_part(out: &mut Vec<u8>, part: &[u8], index: u64) -> Result<(), EncodeError> {
    let isot = u16::try_from(index)
        .map_err(|_| EncodeError::encode("jp2k: a tile index does not fit in Isot"))?;
    let psot = u32::try_from(part.len())
        .map_err(|_| EncodeError::encode("jp2k: a tile-part is too long for Psot"))?;
    let at = out.len();
    out.extend_from_slice(part);
    out[at + 4..at + 6].copy_from_slice(&isot.to_be_bytes());
    out[at + 6..at + 10].copy_from_slice(&psot.to_be_bytes());
    out[at + 10] = 0;
    out[at + 11] = 1;
    Ok(())
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

/// What one component header says about a raster carrier: how many bits, how
/// many bytes it takes in the raster, and whether `Ssiz`'s top bit is set.
///
/// A triple of loose values would let a caller pair the width of one carrier
/// with the sign of another; naming the three fields is what stops the
/// `false` that used to sit in the `ImageComponent::new` call from being
/// re-introduced by a positional argument (issue #905).
#[cfg(feature = "jp2k")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ComponentLayout {
    /// Bits per sample, which becomes `(Ssiz & 0x7f) + 1` in the file.
    precision: u8,
    /// Bytes this carrier takes per sample in the raster.
    element_bytes: usize,
    /// Whether the samples are two's complement, which becomes `Ssiz`'s top
    /// bit.
    signed: bool,
}

/// The component layout for a raster carrier.
///
/// # Errors
///
/// [`EncodeError::Encode`] for the float carriers, which is what
/// `vips jp2ksave` does with a `float` or `double` image: it fails with `not
/// an integer format` rather than quantising behind the caller's back; and
/// for the 32-bit integer carriers, which vips accepts and then fails to
/// read back.
///
/// The signed 8- and 16-bit carriers are **not** refusals. They were, until
/// issue #905, and the split between them and the 32-bit pair is measured
/// rather than assumed: under `--lossless`, `char` and `short` round-trip
/// through vips exactly while `int` and `uint` do not.
#[cfg(feature = "jp2k")]
fn sample_depth(format: PixelFormat) -> Result<ComponentLayout, EncodeError> {
    let layout = |precision, element_bytes, signed| ComponentLayout {
        precision,
        element_bytes,
        signed,
    };
    match format {
        PixelFormat::Gray8 | PixelFormat::Rgb8 | PixelFormat::Rgba8 | PixelFormat::Multi8(_) => {
            Ok(layout(8, 1, false))
        }
        PixelFormat::Gray16
        | PixelFormat::Rgb16
        | PixelFormat::Rgba16
        | PixelFormat::Multi16(_) => Ok(layout(16, 2, false)),
        // JPEG 2000 carries signed samples natively, so these are written
        // rather than refused: `Ssiz` gets its top bit and the samples go in
        // as two's complement. Measured on `/opt/homebrew/bin/vips` 8.18.6,
        // `jp2ksave --lossless` then `jp2kload` on a raster holding
        // `[-5, 100, -100, 7]`: `char` and `short` both come back unchanged,
        // and the files this encoder writes carry the same `Ssiz` bytes vips
        // writes, `0x87` and `0x8f`, and read back through `vips jp2kload`
        // sample for sample (issue #905).
        PixelFormat::Int8(_) => Ok(layout(8, 1, true)),
        PixelFormat::Int16(_) => Ok(layout(16, 2, true)),
        PixelFormat::RgbaF32 | PixelFormat::FloatF32(_) => Err(EncodeError::encode(format!(
            "jp2k: JPEG 2000 stores integer samples and {format:?} is float; cast to an \
             integer format first, so the quantisation is yours rather than the encoder's \
             (vips jp2ksave refuses the same image with `not an integer format`)"
        ))),
        // `jp2ksave` *accepts* a `uint` image and does not round-trip it.
        // Measured on `/opt/homebrew/bin/vips` 8.18.6, saving a `uint`
        // raster and loading it back: 1 reads back as 2147483648, 255 as
        // 2147484160, 65535 as 2147614720 and 70000 as 2147622912, which is
        // 2^31 plus roughly twice the sample. No value survives. So there
        // is nothing to be faithful to here, and a typed refusal is the
        // implementation rather than a gap (issue #517).
        PixelFormat::Uint32(_) | PixelFormat::Int32(_) => Err(EncodeError::encode(format!(
            "jp2k: this encoder writes 8- and 16-bit samples and {format:?} is 32-bit; \
             cast to an 8/16-bit format first (vips jp2ksave accepts a 32-bit image \
             but does not read it back: with --lossless, uint 7 returns as 2147483662 \
             and int 7 returns as 14)"
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

    /// The SHA-256 of a byte slice, for pinning an encoder's whole output
    /// against the oracle's.
    #[cfg(feature = "jp2k")]
    fn digest(bytes: &[u8]) -> String {
        use sha2::Digest;
        crate::hex::hex_lower(&sha2::Sha256::digest(bytes))
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

    /// The samples of a raster read at its own sample kind, so a signed
    /// carrier comes back as the numbers the file holds rather than as their
    /// bit patterns.
    ///
    /// Keyed on [`PixelFormat::kind`] rather than on a byte width, for the
    /// reason issue #607 gives: `Int8` and `Gray8` are both one byte and
    /// mean different things, and `samples` above would read -5 as 251.
    #[cfg(feature = "jp2k")]
    fn signed_samples(raster: &Raster) -> Vec<i32> {
        use crate::pixel::SampleKind;
        let data = raster.data();
        match raster.format().kind() {
            SampleKind::I8 => data.iter().map(|b| i32::from(*b as i8)).collect(),
            SampleKind::I16 => data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| i32::from(i16::from_ne_bytes(*c)))
                .collect(),
            other => panic!("signed_samples wants a signed carrier, not {other:?}"),
        }
    }

    /// Every `#[test]` in this file, as (line, name, doc block, body).
    ///
    /// The body is widened by one level of indirection: any `const` this file
    /// declares whose name the body mentions is appended to it, so a test
    /// that drives a table reaches the fixtures the table names. That is the
    /// same one-hop rule `tests/miri_ignore_convention.rs` follows into a
    /// test helper, and it is what keeps
    /// `the_reversible_fixtures_decode_to_the_bytes_vips_produces` honest:
    /// its doc names three fixtures its body reaches only through `EXACT`.
    fn tests_with_docs_and_bodies(source: &str) -> Vec<(usize, String, String, String)> {
        let lines: Vec<&str> = source.lines().collect();

        // `const NAME: ... ;`, from its first line to the first line that ends
        // the item. Every const in this file is a single item terminated that
        // way, and one that is not simply contributes a shorter block.
        let mut consts: Vec<(String, String)> = Vec::new();
        for (i, line) in lines.iter().enumerate() {
            let trimmed = line.trim_start();
            let Some(rest) = trimmed.strip_prefix("const ") else {
                continue;
            };
            let Some(name) = rest.split(':').next().map(str::trim) else {
                continue;
            };
            let mut end = i;
            while end < lines.len() && !lines[end].trim_end().ends_with(';') {
                end += 1;
            }
            consts.push((
                name.to_string(),
                lines[i..=end.min(lines.len() - 1)].join("\n"),
            ));
        }

        let mut out = Vec::new();
        let mut i = 0;
        while i < lines.len() {
            let Some(name) = lines[i]
                .trim_start()
                .strip_prefix("fn ")
                .and_then(|rest| rest.split('(').next())
            else {
                i += 1;
                continue;
            };
            let mut attrs = i;
            while attrs > 0 && lines[attrs - 1].trim_start().starts_with("#[") {
                attrs -= 1;
            }
            if !lines[attrs..i].iter().any(|l| l.trim() == "#[test]") {
                i += 1;
                continue;
            }
            // The block doc above the attribute stack, if there is one.
            let mut doc = String::new();
            if attrs > 0 && lines[attrs - 1].trim().ends_with("*/") {
                let mut start = attrs - 1;
                while start > 0 && !lines[start].trim_start().starts_with("/**") {
                    start -= 1;
                }
                doc = lines[start..attrs].join("\n");
            }
            // The body, to the closing brace at the function's own indent.
            let indent = &lines[i][..lines[i].len() - lines[i].trim_start().len()];
            let closing = format!("{indent}}}");
            let mut end = i + 1;
            while end < lines.len() && lines[end] != closing {
                end += 1;
            }
            let mut body = lines[i..=end.min(lines.len() - 1)].join("\n");
            for (const_name, text) in &consts {
                if body.contains(const_name.as_str()) {
                    body.push('\n');
                    body.push_str(text);
                }
            }
            out.push((attrs, name.to_string(), doc, body));
            i = end + 1;
        }
        out
    }

    /**
     * Every `#[test]` in this file carries its own doc block, checked by
     * reading the file rather than by convention, because this file has
     * already lost one twice.
     * The block for `the_image_origin_is_the_one_divergence_on_geometry`
     * came off its function when two PRs merged minutes apart (#846 and
     * #855) and their hunks interleaved. Issue #869 filed that, #891 moved
     * the block, and it still did not land on its test: measured on
     * `origin/main` while issue #926 was in flight, the block sat above the
     * *band ceiling* test's own doc block, so that test's rendered doc
     * opened with a paragraph about image origins and the origin test had
     * none. Nothing went red. `cargo fmt --check` is clean, `make clippy`
     * is silent across all nine features, and `cargo doc` with all three
     * lints denied has nothing to say, because for a private test item
     * rustdoc has no opinion about a lost doc and every link still
     * resolves.
     * A doc that drifted onto the wrong function is the failure mode, and
     * the observable half of it is that some other function ends up with
     * none, because a block doc comment always attaches to the item below
     * it. So that is what this asserts, over the file's own text.
     * Input: `src/jp2k.rs` -> Output: no `#[test]` without a doc block of
     * its own, and a count of the tests scanned so a parse that stopped
     * early cannot pass quietly.
     */
    #[test]
    fn every_test_in_this_file_keeps_its_own_doc_block() {
        // `include_str!` rather than `std::fs::read_to_string`, for the same
        // reason the fixtures above are `include_bytes!`: it keeps this off
        // `tests/miri_fs_test_inventory.txt`.
        let source = include_str!("jp2k.rs");
        let tests = tests_with_docs_and_bodies(source);
        let undocumented: Vec<(usize, &str)> = tests
            .iter()
            .filter(|(_, _, doc, _)| doc.is_empty())
            .map(|(line, name, _, _)| (*line, name.as_str()))
            .collect();
        assert!(
            undocumented.is_empty(),
            "these tests have no doc block of their own, which means their block is \
             sitting on somebody else's function: {undocumented:?}"
        );
        // The negative control on the scanner itself: a parse that matched
        // nothing would report an empty offender list too, which is the
        // empty-result trap. This file has dozens of tests, so a number in
        // single figures means the walk stopped early.
        assert!(
            tests.len() > 30,
            "the scanner found only {} tests in this file, so it is not reading what it \
             thinks it is reading",
            tests.len()
        );
    }

    /**
     * And the half the count cannot see: a doc block is on the test it
     * describes, not merely on some test.
     * "Every test has a doc" is satisfied perfectly by a block sitting above
     * the wrong one, which is exactly the state issue #926 found: two blocks
     * stacked on the band-ceiling test and none on the origin test. Position
     * had drifted and content had not, so the content is what identifies the
     * owner. Every doc block in this file names the fixtures its test drives,
     * in an `Input:` line, so the check is that a fixture named in a doc is
     * a fixture that test can reach.
     * The reach is one hop, through a `const` the body names, because
     * `the_reversible_fixtures_decode_to_the_bytes_vips_produces` documents
     * three fixtures it touches only through `EXACT`. Without that hop the
     * check would have a false positive on the day it landed, which is the
     * fastest way to get a guard deleted.
     * Input: `src/jp2k.rs` -> Output: no doc block naming a fixture its own
     * test cannot reach, plus the count of fixture mentions actually
     * checked, so a scan that found no fixture names at all cannot pass as
     * a clean file.
     */
    #[test]
    fn a_doc_block_names_fixtures_the_test_under_it_actually_reaches() {
        let source = include_str!("jp2k.rs");
        // Every committed fixture name, taken from the file's own string
        // literals rather than from a second hand-written list.
        let mut names: Vec<&str> = Vec::new();
        for piece in source.split('"').skip(1).step_by(2) {
            if [".j2k", ".jp2", ".bin"]
                .iter()
                .any(|suffix| piece.ends_with(suffix))
                && !names.contains(&piece)
            {
                names.push(piece);
            }
        }
        assert!(
            names.len() > 20,
            "the fixture scan found only {} names, so it is not reading the file it \
             thinks it is reading",
            names.len()
        );

        let mut checked = 0;
        let mut stranded: Vec<(String, &str)> = Vec::new();
        for (_, test, doc, body) in tests_with_docs_and_bodies(source) {
            for name in &names {
                if doc.contains(name) {
                    checked += 1;
                    if !body.contains(name) {
                        stranded.push((test.clone(), name));
                    }
                }
            }
        }
        assert!(
            stranded.is_empty(),
            "these doc blocks name a fixture their own test never touches, which is what \
             a block sitting on the wrong function looks like: {stranded:?}"
        );
        // The positive control. An empty `stranded` proves nothing unless
        // the scan actually found doc blocks naming fixtures, and this file
        // has dozens.
        assert!(
            checked > 20,
            "only {checked} fixture mentions were checked, so this passed by finding \
             nothing rather than by finding nothing wrong"
        );
    }

    /**
     * Pins the signed decode as the numbers vips reports, not as an offset.
     * `hayro-jpeg2000` DC-level-shifts every component into the unsigned
     * range whatever `SIZ` says, so the loader takes the sign bit off the
     * codestream itself and subtracts the shift back off. Both halves are
     * asserted: the carrier is the signed one, and the five samples are the
     * ones the committed capture records for `vips jp2kload`.
     * The values are the point. `depth12s.j2k` holds `[-2048, -1, 0, 1,
     * 2047]` and vips reports `[-32768, -16, 0, 16, 32752]`, so the row
     * carries the left-justification and the sign together: an
     * implementation that forgot the shift would say `[-2048, -1, 0, 1,
     * 2047]` and one that forgot the sign would say `[0, 32752, -32768,
     * -32752, 65520 as i16]`. Neither can pass this.
     * `depth12u.j2k` is the same file with the sign bit clear and is the
     * positive control that the carrier moved because of that bit and not
     * because of the depth or the geometry.
     * Input: `depth12s.j2k` -> Output: `Int16(1)` holding the five vips
     * numbers, with `bits-per-sample` still 12.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn a_signed_component_decodes_to_the_signed_carrier_vips_uses() {
        let one = std::num::NonZeroU16::new(1).expect("1 is non-zero");
        let raster = decoded("depth12s.j2k");
        assert_eq!(
            raster.format(),
            PixelFormat::Int16(one),
            "a 12-bit signed component is `short` in vips and `Int16` here"
        );
        assert_eq!(
            signed_samples(&raster),
            vec![-32768, -16, 0, 16, 32752],
            "the committed capture records exactly these five for vips jp2kload, and \
             they carry the left-justification and the sign at once"
        );
        assert_eq!(
            int_field(&raster, "bits-per-sample"),
            Some(12),
            "the true depth still travels here and not in the value"
        );
        assert_eq!(
            raster.interpretation(),
            Interpretation::Grey16,
            "vips tags the same file grey16"
        );

        // The control: the same 5x1 shape at the same precision with the sign
        // bit clear is unsigned, so the carrier above followed that bit.
        let control = decoded("depth12u.j2k");
        assert_eq!(control.format(), PixelFormat::Gray16);
        assert_eq!(samples(&control), vec![0, 16, 16384, 65504, 65520]);
    }

    /**
     * Pins the one carrier refusal left on the decode side, and that both
     * 31-bit fixtures now reach it.
     * Before issue #905 `int31.jp2` was refused by the sign bit, which ran
     * first; the sign bit is a carrier now, so the precision ceiling is what
     * catches it, and `uint31.jp2` reaches the same refusal by the same
     * route. That is the assertion: the two files are refused for the same
     * reason, and the reason is the decoder's `f32` container rather than a
     * missing carrier.
     * The control is `depth16u.j2k`, one bit under the ceiling, so a refusal
     * that fired on the shape rather than the number would take it down too.
     * Input: `uint31.jp2`, `int31.jp2` -> Output:
     * `PrecisionNotSupported { precision: 31, max: 16 }` for both.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn thirty_one_bits_is_the_decode_ceiling_whichever_sign_it_carries() {
        for name in ["uint31.jp2", "int31.jp2"] {
            let err = decode_jp2k(fixture(name), DecodeLimits::default())
                .expect_err("31 bits per sample has no carrier");
            let SourceError::Jp2k(Jp2kError::PrecisionNotSupported {
                component,
                precision,
                max,
            }) = err
            else {
                panic!("{name} must be refused by precision: {err:?}");
            };
            assert_eq!(
                (component, precision, max),
                (0, 31, MAX_PRECISION),
                "{name}"
            );
        }
        // The control: one bit under the ceiling still decodes.
        assert_eq!(decoded("depth16u.j2k").format(), PixelFormat::Gray16);
    }

    /// A copy of `bytes` with `Ssiz`'s sign bit set on the named components
    /// and nothing else touched.
    ///
    /// The whole point of building the input this way is that the sample
    /// data, the tile layout, the `colr` box and the geometry are the
    /// committed fixture's, so the only thing under test is the
    /// declaration. It asserts that every named component actually moved,
    /// because a flip that silently failed to apply would leave a test
    /// asserting a refusal that never had anything to refuse.
    #[cfg(feature = "jp2k")]
    fn with_sign_bits(bytes: &[u8], components: &[usize]) -> Vec<u8> {
        let layout = ContainerLayout::parse(bytes).expect("container");
        let mut out = bytes.to_vec();
        let base = layout.codestream;
        let count = usize::from(u16::from_be_bytes([out[base + 40], out[base + 41]]));
        for component in components {
            assert!(*component < count, "component {component} of {count}");
            let at = base + 42 + component * 3;
            assert_eq!(
                out[at] & 0x80,
                0,
                "component {component} is already signed, so setting the bit proves nothing"
            );
            out[at] |= 0x80;
        }
        out
    }

    /**
     * Pins the refusal for a file whose components disagree about the sign
     * bit, and pins it as parity rather than as a gap.
     * A raster carries one sample kind for every band, so there is no
     * carrier for a file that mixes them. vips has the same problem and the
     * same answer: measured on 8.18.6, `rgb_lossless.jp2` with component
     * 1's `Ssiz` sign bit flipped and nothing else touched fails with
     * `jp2kload: components differ in precision`, while the untouched file
     * decodes. So matching the refusal is the faithful thing here.
     * Two controls, and they are what make the row mean something. The
     * untouched fixture still decodes, so the refusal is the flipped bit and
     * not the rewrite; and the *same* fixture with all three bits flipped
     * decodes as `Int8(3)`, so the refusal is the disagreement and not the
     * sign.
     * Input: `rgb_lossless.jp2` with one, none and all three sign bits set
     * -> Output: `MixedComponentSignedness { signed: 1, unsigned: 0 }`, a
     * clean `Rgb8`, and a clean `Int8(3)`.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn components_that_disagree_about_the_sign_bit_are_refused_the_way_vips_refuses_them() {
        let three = std::num::NonZeroU16::new(3).expect("3 is non-zero");
        let base = fixture("rgb_lossless.jp2");

        let mixed = with_sign_bits(base, &[1]);
        let err = decode_jp2k(&mixed, DecodeLimits::default())
            .expect_err("a file whose components disagree has no carrier");
        assert!(
            matches!(
                err,
                SourceError::Jp2k(Jp2kError::MixedComponentSignedness {
                    signed: 1,
                    unsigned: 0
                })
            ),
            "the refusal must name both sides: {err:?}"
        );

        // Control one: the untouched fixture, so the refusal is the bit.
        assert_eq!(decoded("rgb_lossless.jp2").format(), PixelFormat::Rgb8);

        // Control two: the same file with every component signed, so the
        // refusal is the disagreement rather than the sign. vips reads this
        // one as `char` / `srgb` too (measured: `[-128, -128, -128, -67,
        // -31, -99]` for the first two pixels, against `[0, 0, 0, 61, 97,
        // 29]` unsigned, which is the same numbers less 128).
        let all_signed = with_sign_bits(base, &[0, 1, 2]);
        let raster = decode_jp2k(&all_signed, DecodeLimits::default())
            .expect("an all-signed file has a carrier");
        assert_eq!(raster.format(), PixelFormat::Int8(three));
        assert_eq!(
            raster.interpretation(),
            Interpretation::Srgb,
            "vips tags the same file srgb"
        );
        assert_eq!(
            signed_samples(&raster)[..6],
            [-128, -128, -128, -67, -31, -99],
            "the same samples as the unsigned control, less 128, which is what the \
             DC level shift comes to at 8 bits"
        );
    }

    /**
     * Pins the second signed refusal: a signed file in the shape `jp2kload`
     * runs its inverse YCC over is refused rather than reproduced, because
     * vips's answer there is not one a carrier can hold.
     * Measured on 8.18.6 by writing the committed `sub420.j2k` shape signed
     * with `opj_compress` (`-F 8,4,3,8,s@1x1:2x2:2x2 -mct 0`): vips
     * subtracts the YCC offset *inside* the component's own signed carrier,
     * so `-112 - 128` wraps to 16, and then clamps the transform's output to
     * the unsigned range before storing it in a `char`. The red band comes
     * out 0 at every pixel and the blue band wraps past 127 into negatives:
     * `[0, 5, 28]` at pixel 0 and `[0, 28, -122]` at pixel 4, against
     * `[255, 87, 0]` and `[200, 109, 36]` for the same file written
     * unsigned. Matching that would be matching an oracle that has lost the
     * picture.
     * The control is the untouched fixture, which still decodes through the
     * YCC path to the pixels vips produces, so the refusal is the sign bit
     * and not the shape.
     * Input: `sub420.j2k` with all three sign bits set -> Output:
     * `SignedInverseYcc { components: 3 }`, and a clean decode without them.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn a_signed_file_in_the_inverse_ycc_shape_is_refused_rather_than_reproduced() {
        let signed = with_sign_bits(fixture("sub420.j2k"), &[0, 1, 2]);
        let err = decode_jp2k(&signed, DecodeLimits::default())
            .expect_err("vips's answer for this shape is not one a carrier can hold");
        assert!(
            matches!(
                err,
                SourceError::Jp2k(Jp2kError::SignedInverseYcc { components: 3 })
            ),
            "the refusal must name the shape: {err:?}"
        );

        // The control: without the sign bits the same file runs the inverse
        // YCC and lands on vips's pixels, so this refusal is the sign and
        // not the subsampling.
        let raster = decoded("sub420.j2k");
        assert_eq!(raster.format(), PixelFormat::Rgb8);
        assert_eq!(raster.data()[..3], [255, 87, 0]);
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
        // component 8-bit, and for 3 components 16-bit). Every cell decodes,
        // which it did not before: the three that used to be refusals were
        // #771, #848 and #849, so a refusal anywhere here is now a failure
        // rather than a table entry.
        use Interpretation::{Bw, Cmyk, Grey16, Rgb16, Srgb};
        let cells: &[(u32, Interpretation, Interpretation, Interpretation)] = &[
            (12, Cmyk, Cmyk, Cmyk),
            // CIELab, which openjpeg does not recognise, so vips guesses from
            // the band count exactly as it does for the undefined 99.
            (14, Srgb, Bw, Rgb16),
            // sRGB, sYCC and e-YCC on one component are the combination vips
            // gets wrong: it reports `3 bands, srgb` and then cannot read a
            // pixel. This keeps the one real band and takes the tag, which is
            // `a_one_component_file_tagged_srgb_keeps_its_band_and_takes_the_tag`.
            (16, Srgb, Srgb, Rgb16),
            (17, Bw, Bw, Grey16),
            (18, Srgb, Srgb, Rgb16),
            // 20 (ROMM-RGB) and 21 (YPbPr) are registered JPEG 2000 values
            // openjpeg does not map, so vips takes the UNSPECIFIED arm and
            // guesses from the band count. They are in the table because they
            // were the only unrecognised enums `hayro-jpeg2000` read rather
            // than refused, which made them the only cells where the
            // band-count fallback was observable on **one** component.
            (20, Srgb, Bw, Rgb16),
            (21, Srgb, Bw, Rgb16),
            (24, Srgb, Srgb, Rgb16),
            (99, Srgb, Bw, Rgb16),
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
            let cases: [(&str, Vec<u8>, Interpretation); 3] = [
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
                let raster = decode_jp2k(&bytes, DecodeLimits::default())
                    .unwrap_or_else(|e| panic!("EnumCS {enumcs} on {shape} must decode, got {e}"));
                assert_eq!(
                    raster.interpretation(),
                    want,
                    "EnumCS {enumcs} on {shape}: vips reports {want:?}"
                );
                checked += 1;
            }
        }
        assert_eq!(
            checked, 27,
            "the sweep has to reach every cell, and after #771, #848 and #849 every one \
             of them decodes"
        );
    }

    /**
     * A `METH = 2` `colr` box is a profile and never an enumerated colour
     * space, however its first four bytes read (issue #767).
     * The two readers share the same three-byte `colr` prelude, so an
     * enumerated-space reader that checked the length and not the method would
     * take the first word of an ICC profile as an `EnumCS`. A real profile
     * begins with its own size, and a 17-byte one would read as `EnumCS 17`
     * and retag a **three**-band image `b-w`. Nothing in the fixture set
     * happens to land on one of the five recognised values, so the sweep alone
     * leaves that mutation alive; this builds the byte pattern on purpose, on
     * three components, which is where honouring it and ignoring it give
     * different answers.
     * Input: `rgb_lossless.jp2` with its `colr` box switched to `METH = 2` and
     * its payload set to `00 00 00 11` -> Output: `srgb`, the band-count
     * guess, with those four bytes attached as the profile; and the same four
     * bytes under `METH = 1` giving `b-w`, which is the control.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn a_meth_2_profile_is_never_read_as_an_enumerated_colour_space() {
        fn retagged_meth(meth: u8, word: u32) -> Vec<u8> {
            let mut bytes = fixture("rgb_lossless.jp2").to_vec();
            let at = bytes
                .windows(4)
                .position(|w| w == b"colr")
                .expect("a colr box");
            bytes[at + 4] = meth;
            bytes[at + 7..at + 11].copy_from_slice(&word.to_be_bytes());
            bytes
        }

        // 17 is greyscale, the enum whose answer differs from the band-count
        // guess on a three-band file. Under METH = 2 those four bytes are the
        // profile and nothing else.
        let profile = decode_jp2k(&retagged_meth(2, 17), DecodeLimits::default())
            .expect("a METH=2 box decodes");
        assert_eq!(
            profile.interpretation(),
            Interpretation::Srgb,
            "three bands with no usable enum is the band-count guess, not the \
             first word of the profile"
        );
        assert_eq!(
            profile.icc_profile(),
            Some(&[0u8, 0, 0, 17][..]),
            "the four bytes are the profile, unvalidated, the way jp2kload copies it"
        );

        // The control that makes the assertion above mean something: the very
        // same four bytes under METH = 1 *are* an enum and do retag the image.
        let enumerated = decode_jp2k(&retagged_meth(1, 17), DecodeLimits::default())
            .expect("a METH=1 box decodes");
        assert_eq!(enumerated.interpretation(), Interpretation::Bw);
        assert_eq!(enumerated.icc_profile(), None);
    }

    /**
     * A `colr` box the decoder cannot resolve no longer refuses the file
     * (issues #771, #848, #849).
     * Three enumerated colour spaces used to stop the decode dead where
     * `jp2kload` reads the file and hands back pixels: anything openjpeg does
     * not recognise (#771), e-YCC (#848), and CIELab on one component (#849).
     * All three are `hayro-jpeg2000` resolving the `colr` box and refusing what
     * it cannot map, and none of them is a property of the codestream, which
     * decodes perfectly well.
     * Every `want` here is `vips getpoint` on 8.18.6 over the same bytes, and
     * the shape of the fix is that the enum never reaches the decoder at all:
     * `crate::jp2k` reads it for the interpretation (#767) and for the inverse
     * YCC, and hands the decoder a neutral one.
     * Input: `rgb_lossless.jp2` and a wrapped `depth8u.j2k` at the four enums
     * that used to refuse -> Output: a decode, with vips's pixel at (0, 0).
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn a_colr_box_the_decoder_cannot_resolve_no_longer_refuses_the_file() {
        // (enum, vips's three bands at 0,0 on the retagged rgb_lossless.jp2,
        // vips's one band at 0,0 on the wrapped depth8u.j2k).
        let cases: &[(u32, [u8; 3], u8)] = &[
            // Unrecognised, so vips takes UNSPECIFIED and touches nothing.
            (99, [0, 0, 0], 0),
            (20, [0, 0, 0], 0),
            // e-YCC, which vips maps onto the same answer as sYCC, transform
            // included: the 135 in the middle band is the inverse YCC running.
            (24, [0, 135, 0], 0),
            // CIELab, which openjpeg does not recognise either, so vips leaves
            // the samples alone where the decoder used to convert them.
            (14, [0, 0, 0], 0),
        ];

        for (enumcs, three, one) in cases {
            let raster = decode_jp2k(
                &retagged(fixture("rgb_lossless.jp2"), *enumcs),
                DecodeLimits::default(),
            )
            .unwrap_or_else(|e| panic!("EnumCS {enumcs} on three components must decode: {e}"));
            assert_eq!(
                &raster.data()[..3],
                three,
                "EnumCS {enumcs}: vips reads {three:?} at (0, 0)"
            );

            let raster = decode_jp2k(
                &wrapped(fixture("depth8u.j2k"), 5, 1, 1, 8, *enumcs),
                DecodeLimits::default(),
            )
            .unwrap_or_else(|e| panic!("EnumCS {enumcs} on one component must decode: {e}"));
            assert_eq!(
                raster.data()[0],
                *one,
                "EnumCS {enumcs}: vips reads {one} at (0, 0)"
            );
        }
    }

    /**
     * The inverse YCC follows the `colr` box's enum and the subsampling
     * together, and the pixels stay vips's (issues #848, #771).
     * This is the half of the fix that could go wrong quietly. The enum stops
     * reaching the decoder, so `hayro-jpeg2000` stops running its own sYCC
     * transform, and this module has to run it instead or a subsampled sYCC
     * file comes back untransformed. The condition is openjpeg's: sYCC or
     * e-YCC by enum, or an unspecified colour space over subsampled chroma.
     * Every `want` is `vips getpoint FILE 0 0` on 8.18.6. The three lossy
     * fixtures carry the tolerance their own pins carry, because the 9/7
     * wavelet is float-specified; the tolerance is nowhere near wide enough to
     * hide the failure this guards, since the same file read untransformed
     * comes back at 30 in the first band against vips's 4.
     * Input: the five container shapes that bracket the rule -> Output:
     * transformed where vips transforms and not where it does not.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn the_inverse_ycc_follows_the_enum_and_the_subsampling_together() {
        fn close(got: &[u8], want: [u8; 3], tolerance: i32, what: &str) {
            for (band, want) in want.iter().enumerate() {
                let delta = i32::from(got[band]) - i32::from(*want);
                assert!(
                    delta.abs() <= tolerance,
                    "{what}: band {band} is {} where vips reads {want}, off by {delta} \
                     against a tolerance of {tolerance}",
                    got[band]
                );
            }
        }

        // A bare codestream with subsampled chroma: unspecified colour space,
        // so the SIZ heuristic turns it on. Reversible, so this one is exact.
        assert_eq!(&decoded("sub420.j2k").data()[..3], &[255, 87, 0]);

        // The same subsampling inside a JP2 tagged sYCC, which is where the
        // enum has to carry the decision now.
        close(
            decoded("chroma_sub_on.jp2").data(),
            [4, 1, 241],
            3,
            "chroma_sub_on.jp2",
        );
        close(
            decoded("chroma_tiny_sub_on.jp2").data(),
            [75, 75, 75],
            3,
            "chroma_tiny_sub_on.jp2",
        );

        // A JP2 tagged sRGB with no subsampling: no transform either way.
        close(
            decoded("chroma_sub_off.jp2").data(),
            [0, 1, 255],
            3,
            "chroma_sub_off.jp2",
        );

        // An unsubsampled JP2 tagged sYCC: the enum alone turns it on, which
        // the container shape cannot say. Lossless fixture, so exact.
        let sycc = decode_jp2k(
            &retagged(fixture("rgb_lossless.jp2"), 18),
            DecodeLimits::default(),
        )
        .expect("decodes");
        assert_eq!(&sycc.data()[..3], &[0, 135, 0]);

        // And the control that says the enum is doing it rather than the retag
        // machinery: the same file at sRGB is untouched.
        let srgb = decode_jp2k(
            &retagged(fixture("rgb_lossless.jp2"), 16),
            DecodeLimits::default(),
        )
        .expect("decodes");
        assert_eq!(&srgb.data()[..3], &[0, 0, 0]);

        // The two rows that pin the *fallback*, both measured on 8.18.6 by
        // rewriting nothing but `chroma_sub_on.jp2`'s `colr` box. They are the
        // pair the old `bare && subsampled` condition could not tell apart,
        // and it got the second one wrong.
        //
        //   METH = 1, sRGB      vips reads `29 248 110`, no transform
        //   METH = 2, a profile vips reads `4 1 241`, transform
        //
        // A profile box leaves the colour space exactly where `SIZ` put it, so
        // the subsampling still decides; a recognised non-YCC enum overrides
        // it. Nothing about the codestream differs between them.
        let mut as_srgb = fixture("chroma_sub_on.jp2").to_vec();
        let at = as_srgb
            .windows(4)
            .position(|w| w == b"colr")
            .expect("a colr box");
        as_srgb[at + 7..at + 11].copy_from_slice(&16u32.to_be_bytes());
        close(
            decode_jp2k(&as_srgb, DecodeLimits::default())
                .expect("decodes")
                .data(),
            [29, 248, 110],
            3,
            "chroma_sub_on.jp2 retagged sRGB",
        );

        let mut as_profile = fixture("chroma_sub_on.jp2").to_vec();
        as_profile[at + 4] = 2; // METH = 2, so the payload is a profile
        close(
            decode_jp2k(&as_profile, DecodeLimits::default())
                .expect("decodes")
                .data(),
            [4, 1, 241],
            3,
            "chroma_sub_on.jp2 as a METH=2 profile box",
        );
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

    /**
     * Pins the wall four components used to stand against, and corrects the
     * table that was published for it (issue #769).
     * The earlier sweep asked `hayro-jpeg2000` to read 2, 5 and 6 component
     * codestreams under every `colr` box that could change its mind, and
     * concluded that six components were readable under nothing and that the
     * decoder never reports `ColorSpace::Unknown { num_channels }`. Both
     * conclusions were wrong, and they were wrong because the sweep had no
     * column for the case that matters: **no `colr` box at all**. Its "bare"
     * column was the raw codestream, and the decoder synthesises an sRGB box
     * for that before it validates anything.
     *
     * | components | sRGB | greyscale | CMYK | sYCC | bare codestream | JP2, no colr box |
     * |---|---|---|---|---|---|---|
     * | 2 | refused | Gray + alpha | refused | refused | Gray + alpha | `Unknown { 2 }` |
     * | 5 | refused | refused | **CMYK + alpha** | refused | refused | **`Unknown { 5 }`** |
     * | 6 | refused | refused | refused | refused | refused | **`Unknown { 6 }`** |
     *
     * The last column is the new one and every cell in it was measured; the
     * five before it are the earlier table's, re-run here rather than
     * inherited. So the wall was the box, the decoder has had the arm all
     * along, and `ContainerLayout::unspecified_rewrite` is what reaches it.
     * Input: 2, 5 and 6 component codestreams under six box configurations
     * -> Output: the table above.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn the_wall_above_four_components_was_the_colr_box_and_not_the_count() {
        use hayro_jpeg2000::{DecodeSettings, Image as Jp2kImage};
        use openjpeg2_pure::{EncodeOptions, Encoder, Format, Image, ImageComponent};

        /// A `components`-component 8x6 codestream, written through the same
        /// encoder `Raster::encode_jp2k` uses.
        fn encoded(components: usize, format: Format) -> Vec<u8> {
            let comps: Vec<ImageComponent> = (0..components)
                .map(|band| {
                    ImageComponent::new(
                        8,
                        6,
                        8,
                        false,
                        (0..8u32 * 6)
                            .map(|i| (i as i32 + band as i32) % 251)
                            .collect(),
                    )
                    .expect("a component")
                })
                .collect();
            let image = Image::new(8, 6, openjpeg2_pure::ColorSpace::Srgb, comps)
                .expect("an image the encoder accepts at any band count");
            Encoder::encode(
                &image,
                &EncodeOptions {
                    format,
                    ..EncodeOptions::default()
                },
            )
            .expect("the encoder writes any band count")
        }

        /// What the decoder makes of `bytes`, as the table spells it.
        fn read(bytes: &[u8]) -> Option<(u8, bool)> {
            Jp2kImage::new(bytes, &DecodeSettings::default())
                .ok()
                .map(|image| (image.color_space().num_channels(), image.has_alpha()))
        }

        for (components, cmyk, no_box) in [
            (2usize, None, Some((2u8, false))),
            (5, Some((4u8, true)), Some((5, false))),
            (6, None, Some((6, false))),
        ] {
            let jp2 = encoded(components, Format::Jp2);
            // The five columns the earlier table had.
            assert_eq!(
                read(&retagged(&jp2, ENUMCS_CMYK)),
                cmyk,
                "{components} components under a CMYK box"
            );
            for enumcs in [ENUMCS_SRGB, ENUMCS_GREY, ENUMCS_SYCC] {
                // Greyscale over two components is the one cell here that
                // reads: the decoder's non-strict repair takes the extra
                // component for an opacity. sRGB over two is not, because
                // three plus alpha is four and the repair only reaches one
                // channel up.
                let expected = (components == 2 && enumcs == ENUMCS_GREY).then_some((1u8, true));
                assert_eq!(
                    read(&retagged(&jp2, enumcs)),
                    expected,
                    "{components} components under EnumCS {enumcs}"
                );
            }
            let bare = encoded(components, Format::J2k);
            assert_eq!(
                read(&bare).is_some(),
                components == 2,
                "{components} components as a bare codestream, where the decoder \
                 synthesises its own sRGB box before validating"
            );

            // The column the earlier table did not have.
            let layout = ContainerLayout::parse(&jp2).expect("the encoder's container");
            let stripped = layout
                .unspecified_rewrite(&jp2)
                .expect("a JP2 with a colr box has a removal");
            assert_eq!(
                read(&stripped),
                no_box,
                "{components} components with no colr box at all, which is the arm \
                 #769 turned out to need"
            );
        }
    }

    /**
     * The encoder knobs `openjpeg2-pure-rs` exposes, named exhaustively, so
     * a seventh one announces itself the day upstream adds it (#768).
     * The history is worth keeping because two readings of it were published
     * and both were wrong. #768 said `cp_tdx` / `cp_tdy` and `tile_size_on`
     * are `pub(crate)`; they are `pub`, on a `pub` struct. A correction then
     * said they therefore live in `pub mod openjpeg` and are reachable by
     * hand-rolling `Image::to_opj`; the **module** is `pub(crate)`, so
     * nothing in it is reachable at all. A compile probe against 0.1.1
     * settles it in one line, with the public surface beside it as the
     * control that the probe can reach anything:
     *
     * ```text
     * let _ok = openjp2::EncodeOptions::default();          // compiles
     * let _p = openjp2::openjpeg::opj_cparameters::default();
     * error[E0603]: module `openjpeg` is private
     *  --> openjpeg2-pure-rs-0.1.1/src/lib.rs:72:1
     * ```
     *
     * So there is no encoder knob for the tile grid and there is no route to
     * one either, which is why `encode` reaches the tiled codestream through
     * the format instead. This literal is what says the six have not become
     * seven: `EncodeOptions` is not `#[non_exhaustive]`, so a new field stops
     * it compiling.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn the_encoder_knobs_upstream_exposes_are_still_these_six() {
        use openjpeg2_pure::{EncodeOptions, Format};

        let all_of_them = EncodeOptions {
            format: Format::Jp2,
            threads: 0,
            irreversible: false,
            use_mct: false,
            rates: vec![0.0],
            num_resolutions: None,
        };
        // Naming every field is the assertion; the runtime half just keeps the
        // value alive and pins the two this crate actually sets.
        assert!(!all_of_them.irreversible);
        assert_eq!(all_of_them.rates.len(), 1);
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
     * Pins the rule that replaced "bare codestream and subsampled" as the
     * inverse-YCC condition, at the level of the two predicates rather than
     * the pixels (issues #771, #848).
     * `layout.bare` used to be the whole of the "unspecified colour space"
     * test, on the reasoning that a JP2 always carries a `colr` box. That is
     * true and it is not the same question: a `METH = 2` profile box carries
     * no colour space either, and vips transforms such a file (measured, see
     * `ContainerLayout::runs_inverse_ycc`). The two predicates are asserted
     * here directly because the pixel-level test cannot tell "the decoder did
     * it" from "this module did it", and getting that split wrong applies the
     * transform twice or not at all.
     * Input: the same subsampled codestream in four containers -> Output: who
     * does the transform in each.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn who_runs_the_inverse_ycc_follows_the_colr_box() {
        fn layout_and_header(bytes: &[u8]) -> (ContainerLayout, CodestreamHeader) {
            let layout = ContainerLayout::parse(bytes).expect("container");
            let header = CodestreamHeader::parse(&bytes[layout.codestream..]).expect("codestream");
            (layout, header)
        }

        // A bare codestream: no box, so `SIZ` decides and this module runs it.
        let (layout, header) = layout_and_header(fixture("sub420.j2k"));
        assert_eq!(layout.enum_cs, None);
        assert!(header.chroma_subsampled());
        assert!(layout.decoder_resolves_this_box(&header));
        assert!(layout.runs_inverse_ycc(&header));

        // The same subsampling under a sYCC box: the decoder handles that one,
        // so its bytes go through untouched and this module must NOT transform
        // as well, or the file gets it twice.
        let (layout, header) = layout_and_header(fixture("chroma_sub_on.jp2"));
        assert_eq!(layout.enum_cs, Some(18));
        assert!(header.chroma_subsampled());
        assert!(layout.decoder_resolves_this_box(&header));
        assert!(!layout.runs_inverse_ycc(&header));

        // e-YCC means the same thing to vips and the decoder refuses it, so
        // the box is rewritten and the transform becomes this module's job.
        let eycc = retagged(fixture("chroma_sub_on.jp2"), 24);
        let (layout, header) = layout_and_header(&eycc);
        assert!(!layout.decoder_resolves_this_box(&header));
        assert!(layout.runs_inverse_ycc(&header));

        // A recognised non-YCC enum suppresses the subsampling heuristic
        // entirely: nobody transforms, even though `SIZ` looks like YCC.
        let srgb = retagged(fixture("chroma_sub_on.jp2"), 16);
        let (layout, header) = layout_and_header(&srgb);
        assert!(header.chroma_subsampled());
        assert!(layout.decoder_resolves_this_box(&header));
        assert!(!layout.runs_inverse_ycc(&header));

        // And an unrecognised one does not, so `SIZ` decides again. This is
        // the pair that says the enum is consulted rather than merely present.
        let unknown = retagged(fixture("chroma_sub_on.jp2"), 99);
        let (layout, header) = layout_and_header(&unknown);
        assert!(!layout.decoder_resolves_this_box(&header));
        assert!(layout.runs_inverse_ycc(&header));
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

        // 777 bytes of raster, plus what `hayro-jpeg2000` holds beside it
        // (issue #944). One band, so the per-image term is 777 * 8 and the
        // per-tile term is the 8x8 grid this fixture declares, 64 * 10:
        // 777 + 6216 + 640 = 7633.
        let err = decode_jp2k(bytes, DecodeLimits::default().with_max_alloc_bytes(7632))
            .expect_err("7633 bytes is past a 7632-byte budget");
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
        assert_eq!((needed_bytes, max_alloc_bytes), (7633, 7632));
    }

    // -----------------------------------------------------------------------
    // The encoder
    // -----------------------------------------------------------------------

    /// The band counts every carrier sweep runs over.
    ///
    /// Not `1..=MAX_BANDS`, which was the same thing while the ceiling was 4
    /// and is 16384 since #769. Each count is here for a reason: 1, 3 and 4
    /// reach the named [`PixelFormat`] variants, 2 reaches the multiband one
    /// and is where vips's own band-count guess splits (measured: 2 bands
    /// read back `b-w`, 3 read back `srgb`), and 5 and 6 are the counts the
    /// loader used to refuse, so they are what says the lift reaches the
    /// round trip and not only the encoder.
    #[cfg(feature = "jp2k")]
    const SWEEP_BANDS: [usize; 6] = [1, 2, 3, 4, 5, 6];

    /// A `width` x `height` raster in `format` whose samples are a
    /// per-band ramp, so a band that ends up in the wrong place is visible.
    #[cfg(feature = "jp2k")]
    fn ramp(width: u32, height: u32, format: PixelFormat) -> Raster {
        // Scoped here rather than at module level: `ramp` is the only user and it
        // is `cfg(feature = "jp2k")`, so a module-level import is unused without
        // that feature and `-D warnings` refuses it.
        use crate::pixel::SampleKind;

        let bands = format.channels();
        let kind = format.kind();
        let mut data =
            Vec::with_capacity(width as usize * height as usize * format.bytes_per_pixel());
        for i in 0..(width as usize * height as usize) {
            for band in 0..bands {
                let value = (i * 7 + band * 40) as u32;
                // Keyed on the sample kind, not on its byte width. A width cannot
                // tell `U32` from `F32` or `I32`, so the old `== 2` form wrote one
                // byte per sample for every kind wider than two and produced a
                // fixture at the wrong stride (issue #607).
                match kind {
                    SampleKind::U8 => data.push((value % 256) as u8),
                    SampleKind::I8 => data.push(((value % 256) as i8).to_ne_bytes()[0]),
                    SampleKind::U16 => {
                        data.extend_from_slice(&((value * 271 % 65536) as u16).to_ne_bytes());
                    }
                    // The full 16-bit range and not the non-negative half of
                    // it. `% 32768` would have made every `Int16` sample
                    // positive, and a signed round trip whose fixture never
                    // goes below zero passes just as well against an encoder
                    // that writes the component unsigned, which is the
                    // defect issue #905 closed.
                    SampleKind::I16 => {
                        data.extend_from_slice(
                            &((value * 271 % 65536) as u16 as i16).to_ne_bytes(),
                        );
                    }
                    SampleKind::U32 => data.extend_from_slice(&(value * 271).to_ne_bytes()),
                    SampleKind::I32 => {
                        data.extend_from_slice(&((value * 271) as i32).to_ne_bytes());
                    }
                    SampleKind::F32 => {
                        data.extend_from_slice(&(value as f32).to_ne_bytes());
                    }
                }
            }
        }
        Raster::new(width, height, format, data).expect("ramp fixture")
    }

    /**
     * Pins the lossless encoder as a true round trip at every carrier this
     * codec reads: what goes in comes back out, sample for sample, at every
     * count in `SWEEP_BANDS`, at both element widths, and at both signs.
     * The multiband rows are not padding. A 2-band raster is where vips's own
     * band-count guess splits (measured: 2 bands read back `b-w`, 3 read back
     * `srgb`), and a de-interleaver that transposed bands would still
     * round-trip a 1-band image, so the wide rows are what makes the plane
     * ordering an assertion.
     * The signed rows carry a guard of their own, because they are the ones
     * that can pass vacuously: an all-positive fixture round-trips just as
     * happily through an encoder that never sets `Ssiz`'s sign bit, so each
     * signed row first asserts its fixture holds samples of both signs
     * (issue #905).
     * The five- and six-band rows are the ones #769 lifted, and they are in
     * the same sweep as the rest rather than in a test of their own, because
     * "a band count the loader reads" is one property and not two.
     * Input: four sample kinds at every count in `SWEEP_BANDS`, generated
     * from the kinds rather than listed -> Output: the same pixels back, at
     * the same carrier, with the container always a JP2 whatever the raster.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn every_carrier_survives_a_lossless_round_trip_through_this_crate() {
        use crate::pixel::SampleKind;

        // Generated from the kinds this codec carries rather than listed, so
        // a carrier added to `PixelFormat` cannot slip past the sweep the way
        // three did when issue #516 landed and every hand-written array
        // stopped at the previous last variant. The band counts are
        // [`SWEEP_BANDS`], which is not `1..=MAX_BANDS`: that ceiling is
        // 16384 since #769 and a 16384-band sweep is a different test.
        let carriers: Vec<PixelFormat> = [
            SampleKind::U8,
            SampleKind::U16,
            SampleKind::I8,
            SampleKind::I16,
        ]
        .into_iter()
        .flat_map(|kind| {
            SWEEP_BANDS.into_iter().map(move |bands| {
                PixelFormat::with_kind(bands, kind).expect("every swept band count has a carrier")
            })
        })
        .collect();
        assert_eq!(
            carriers.len(),
            4 * SWEEP_BANDS.len(),
            "four kinds at every swept band count, and the count is spelled out so a \
             kind dropped from the list is a failure rather than a shorter sweep"
        );

        for format in carriers {
            let source = ramp(8, 6, format);
            // The fixture has to be able to catch the bug. A signed round
            // trip whose samples are all non-negative passes against an
            // encoder that writes the component unsigned, so every signed
            // row asserts its own fixture straddles zero before it asserts
            // anything about the codec.
            if matches!(format.kind(), SampleKind::I8 | SampleKind::I16) {
                let values = signed_samples(&source);
                assert!(
                    values.iter().any(|v| *v < 0) && values.iter().any(|v| *v > 0),
                    "{format:?}: the fixture must hold samples of both signs, or this \
                     row cannot tell a signed encode from an unsigned one"
                );
            }
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

    /// The `Ssiz` byte the encoder wrote for every component of `bytes`.
    ///
    /// Read back out of the file rather than off the raster, so the
    /// assertion is about what a third-party reader will see and not about
    /// what this crate meant.
    #[cfg(feature = "jp2k")]
    fn written_ssiz(bytes: &[u8]) -> Vec<u8> {
        let layout = ContainerLayout::parse(bytes).expect("container");
        let codestream = &bytes[layout.codestream..];
        // `SIZ` starts at offset 2 and its component triples at offset 42,
        // which is the same arithmetic `CodestreamHeader::parse` walks; the
        // count comes from `Csiz` at 40.
        let count = usize::from(u16::from_be_bytes([codestream[40], codestream[41]]));
        (0..count).map(|i| codestream[42 + i * 3]).collect()
    }

    /**
     * Pins the encoder's own bytes, not just its round trip: the `Ssiz`
     * byte this crate writes for each carrier is the byte `vips jp2ksave`
     * writes for the libvips band format that carries the same samples.
     * This is the assertion the round trip cannot make. Encoding and
     * decoding through one crate agrees with itself whatever convention it
     * picked, so a codec that wrote every component unsigned and read every
     * component unsigned would still round-trip perfectly and would still be
     * a file nothing else can read. The sign bit in `Ssiz` is what a third
     * party sees, and these four values are what vips puts there.
     * Measured on `/opt/homebrew/bin/vips` 8.18.6: saving a 4x1 raster of
     * `[-5, 100, -100, 7]` as `uchar`, `char`, `ushort` and `short` with
     * `jp2ksave --lossless` gives `Ssiz` of `0x07`, `0x87`, `0x0f` and
     * `0x8f`, and reading the files *this* encoder writes back through
     * `vips jp2kload` returns the same four sample sets exactly.
     * Input: four 4x1 rasters -> Output: `Ssiz` per component, and the
     * precision half of the byte as the control that the sign bit is the
     * only thing that moved.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn the_sign_bit_this_encoder_writes_is_the_one_vips_writes() {
        let one = std::num::NonZeroU16::new(1).expect("1 is non-zero");
        let three = std::num::NonZeroU16::new(3).expect("3 is non-zero");
        let cases: [(PixelFormat, u8); 4] = [
            (PixelFormat::Gray8, 0x07),
            (PixelFormat::Int8(one), 0x87),
            (PixelFormat::Gray16, 0x0f),
            (PixelFormat::Int16(one), 0x8f),
        ];
        for (format, want) in cases {
            let bytes = ramp(4, 1, format)
                .encode_jp2k(SaveOptions::default())
                .unwrap_or_else(|e| panic!("{format:?} must encode: {e}"));
            assert_eq!(
                written_ssiz(&bytes),
                vec![want],
                "{format:?}: vips writes {want:#04x} for the band format that carries \
                 these samples, so anything else is a file vips reads as the wrong sign"
            );
        }

        // Every component gets the bit, not just the first: a loop that set
        // the sign on component 0 and left the rest unsigned would pass the
        // one-band rows above and write a file whose green and blue planes
        // are offset by half the range.
        let bytes = ramp(4, 1, PixelFormat::Int8(three))
            .encode_jp2k(SaveOptions::default())
            .expect("a three-band signed raster must encode");
        assert_eq!(written_ssiz(&bytes), vec![0x87, 0x87, 0x87]);
    }

    /**
     * Pins the exact table issue #905 was filed with, through this crate
     * rather than through vips: `[-5, 100, -100, 7]` survives a lossless
     * round trip on the signed 8- and 16-bit carriers and is refused on the
     * 32-bit ones.
     * The four values are chosen and not a ramp. Two are negative, one of
     * them (-100) is outside the range a naive `as u8` would leave
     * recognisable, and 251 is what -5 would come back as if the samples
     * were widened through the unsigned type. A test whose fixture was
     * `[0, 100, 7]` would pass against exactly that bug.
     * The 32-bit rows are the other half of the split and the reason this
     * issue is not "signed is unsupported": vips writes them and cannot read
     * them back, so refusing is the implementation. Their message has to say
     * 32-bit and must not say signed, or the two refusals are
     * indistinguishable to a caller.
     * Input: `[-5, 100, -100, 7]` at `Int8(1)`, `Int16(1)`, `Int32(1)` and
     * `Uint32(1)` -> Output: an identity for the first two and a typed
     * refusal naming 32-bit for the last two.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn the_values_from_the_issue_round_trip_on_the_signed_carriers() {
        let one = std::num::NonZeroU16::new(1).expect("1 is non-zero");
        let values: [i32; 4] = [-5, 100, -100, 7];

        let bytes: Vec<u8> = values.iter().map(|v| (*v as i8).to_ne_bytes()[0]).collect();
        let source = Raster::new(4, 1, PixelFormat::Int8(one), bytes).expect("Int8 fixture");
        let encoded = source
            .encode_jp2k(SaveOptions::default())
            .expect("Int8 must encode");
        let back = decode_jp2k(&encoded, DecodeLimits::default()).expect("Int8 must decode");
        assert_eq!(back.format(), PixelFormat::Int8(one));
        assert_eq!(
            signed_samples(&back),
            values.to_vec(),
            "vips round-trips these four through `char` exactly, and so does this"
        );
        assert_eq!(
            back.data(),
            source.data(),
            "the reversible 5/3 wavelet is exact, so a signed round trip is an identity \
             on the bytes and not only on the numbers"
        );
        // The same four bytes read as unsigned are `[251, 100, 156, 7]`, and
        // 251 is exactly what -5 comes back as when the sample is widened
        // through `u8` on the way into the encoder. The bytes cannot tell the
        // two apart; the carrier can, which is why `format()` is asserted
        // above and why this row is here rather than a bit-pattern check.
        assert_eq!(samples(&back), vec![251, 100, 156, 7]);

        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| (*v as i16).to_ne_bytes())
            .collect();
        let source = Raster::new(4, 1, PixelFormat::Int16(one), bytes).expect("Int16 fixture");
        let encoded = source
            .encode_jp2k(SaveOptions::default())
            .expect("Int16 must encode");
        let back = decode_jp2k(&encoded, DecodeLimits::default()).expect("Int16 must decode");
        assert_eq!(back.format(), PixelFormat::Int16(one));
        assert_eq!(signed_samples(&back), values.to_vec());

        // The other arm of the split, and it is a different refusal for a
        // different reason: vips accepts a 32-bit image and does not read it
        // back, so there is nothing to be faithful to.
        for format in [PixelFormat::Int32(one), PixelFormat::Uint32(one)] {
            let message = ramp(4, 1, format)
                .encode_jp2k(SaveOptions::default())
                .expect_err("a 32-bit raster has no component this encoder writes")
                .to_string();
            assert!(
                message.contains("32-bit") && !message.contains("signed"),
                "{format:?}: the 32-bit refusal must name the width and must not read as \
                 the signed one, which is no longer a refusal at all: {message}"
            );
        }
    }

    /**
     * A direct sweep of `sample_depth` over every carrier, because the
     * callers cannot reach all of its arms in a way that distinguishes them:
     * `encode` only ever asks for the layout it then uses, so a carrier
     * given the wrong precision or the wrong sign would show up as a wrong
     * file rather than as a wrong answer here, and the two 32-bit arms are
     * refusals no round-trip row can cover.
     * Generated from the sample kinds rather than listed, for the reason
     * issue #516 gives: three carriers arrived at once and four hand-written
     * arrays stopped at the previous last variant. The controls are the
     * collisions the bug would exploit, `Int8` against `Gray8` and `Int16`
     * against `Gray16`, which share a byte width and differ only here.
     * Input: every `PixelFormat` this codec can be handed -> Output: the
     * precision, the element width and the sign bit for the four it writes,
     * and a refusal for the rest.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn sample_depth_answers_precision_width_and_sign_for_every_carrier() {
        use crate::pixel::SampleKind;

        let expected = |kind: SampleKind| -> Option<ComponentLayout> {
            Some(match kind {
                SampleKind::U8 => ComponentLayout {
                    precision: 8,
                    element_bytes: 1,
                    signed: false,
                },
                SampleKind::U16 => ComponentLayout {
                    precision: 16,
                    element_bytes: 2,
                    signed: false,
                },
                SampleKind::I8 => ComponentLayout {
                    precision: 8,
                    element_bytes: 1,
                    signed: true,
                },
                SampleKind::I16 => ComponentLayout {
                    precision: 16,
                    element_bytes: 2,
                    signed: true,
                },
                SampleKind::U32 | SampleKind::I32 | SampleKind::F32 => return None,
            })
        };

        let kinds = [
            SampleKind::U8,
            SampleKind::U16,
            SampleKind::I8,
            SampleKind::I16,
            SampleKind::U32,
            SampleKind::I32,
            SampleKind::F32,
        ];
        let mut written = 0;
        let mut refused = 0;
        for kind in kinds {
            for bands in SWEEP_BANDS {
                let format = PixelFormat::with_kind(bands, kind).expect("every swept band count");
                match (sample_depth(format), expected(kind)) {
                    (Ok(got), Some(want)) => {
                        assert_eq!(got, want, "{format:?}");
                        written += 1;
                    }
                    (Err(e), None) => {
                        refused += 1;
                        let _ = e;
                    }
                    (got, want) => panic!("{format:?}: got {got:?}, wanted {want:?}"),
                }
            }
        }
        assert_eq!(
            (written, refused),
            (4 * SWEEP_BANDS.len(), 3 * SWEEP_BANDS.len()),
            "four carriers written and three refused, at every swept band count; the \
             counts are spelled out so a kind that silently changed sides is a failure"
        );

        // The collision the width-keyed reading gets wrong, stated on its
        // own: `Int8` and `Gray8` are both one byte and only one of them is
        // signed.
        let one = std::num::NonZeroU16::new(1).expect("1 is non-zero");
        let signed8 = sample_depth(PixelFormat::Int8(one)).expect("Int8 is written");
        let unsigned8 = sample_depth(PixelFormat::Gray8).expect("Gray8 is written");
        assert_eq!(
            (signed8.precision, signed8.element_bytes),
            (unsigned8.precision, unsigned8.element_bytes),
            "the two carriers agree on everything a width can see"
        );
        assert_ne!(
            signed8.signed, unsigned8.signed,
            "and disagree on the one thing a width cannot"
        );
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
            .encode_jp2k(SaveOptions::default().with_compression(Compression::Lossy { ratio }))
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
     * Pins the band ceiling as the *format's* now, and pins that the counts
     * the loader used to refuse round-trip through this crate (issue #769).
     * The ceiling was four and the reason was the loader: `hayro-jpeg2000`
     * refuses a component set it cannot map onto greyscale, RGB, CMYK or one
     * of those plus alpha, so anything wider was a file this crate could
     * write and could not read back. Measured, that is a property of the
     * `colr` box and not of the decoder, so `decode` hands it the same file
     * with nothing saying what its colour space is and the ceiling moves to
     * `Csiz`.
     * Three assertions, and the third is the one that would have caught a
     * decoder that merely stopped erroring: the samples come back identical
     * and the interpretation is `srgb`, which is what `vips jp2kload` reports
     * for the same five-, six- and eight-band files (measured, bit for bit).
     * The refusal is still pinned, one component above `MAX_BANDS`, where
     * `openjpeg2-pure-rs` itself answers `EncoderSetupFailed`.
     * Input: 5, 6 and 8 bands at both element widths, and `MAX_BANDS + 1` ->
     * Output: round trips for the first three, a typed refusal for the last.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn the_band_counts_the_loader_used_to_refuse_now_round_trip() {
        for bands in [5usize, 6, 8] {
            for (kind, wide) in [(SampleKind::U8, false), (SampleKind::U16, true)] {
                let format = PixelFormat::with_kind(bands, kind).expect("a multiband carrier");
                let source = ramp(9, 7, format);
                let bytes = source
                    .encode_jp2k(SaveOptions::default())
                    .unwrap_or_else(|e| panic!("{bands} bands at {kind:?}: {e}"));
                let back = decode_jp2k(&bytes, DecodeLimits::default())
                    .unwrap_or_else(|e| panic!("{bands} bands at {kind:?} decode: {e}"));
                assert_eq!(back.format(), format, "{bands} bands at {kind:?}");
                assert_eq!(
                    back.data(),
                    source.data(),
                    "{bands} bands at {kind:?}: the round trip has to be exact, not merely \
                     unrefused"
                );
                // vips's own answer for the same file, and the arm
                // `interpretation` takes for `Unknown { num_channels }`.
                assert_eq!(
                    back.interpretation(),
                    if wide {
                        Interpretation::Rgb16
                    } else {
                        Interpretation::Srgb
                    },
                    "{bands} bands at {kind:?}"
                );
            }
        }

        // The ceiling itself, one past `Csiz`'s range. Measured on
        // `openjpeg2-pure-rs`: 16384 components encode and 16385 come back
        // `EncoderSetupFailed`, so the refusal is where the encoder's is.
        let over = u16::try_from(MAX_BANDS + 1).expect("16385 fits in a u16");
        let over = std::num::NonZeroU16::new(over).expect("16385 is non-zero");
        let err = ramp(1, 1, PixelFormat::Multi8(over))
            .encode_jp2k(SaveOptions::default())
            .expect_err("one component past Csiz");
        let message = err.to_string();
        assert!(matches!(err, EncodeError::Encode(_)), "{err:?}");
        assert!(
            message.contains(&MAX_BANDS.to_string()) && message.contains(&over.to_string()),
            "the refusal must name the ceiling and the count: {message}"
        );

        // The control: the old ceiling still round-trips, so the lift is a
        // widening rather than a re-route.
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
     * Pins that the rewrite lifting the band ceiling fires only on the files
     * the decoder actually refuses, which is the whole of what makes it safe
     * (issue #769).
     * `hayro-jpeg2000` reads a five-component file tagged CMYK with a `cdef`
     * box marking its last channel as an opacity: five is four colour
     * channels and an alpha, a count it can name. Removing the `colr` box
     * there does not widen anything, it **breaks** the file, because
     * `Unknown { num_channels: 5 }` plus an alpha is six channels against
     * five components and the decoder gives up. So a rewrite keyed on the
     * component count rather than on the refusal would turn a file that
     * decodes into one that does not.
     * The positive control is the third assertion, and without it the first
     * two could both pass against a rewrite that never runs: the same file
     * with its box actually removed is handed to the decoder directly and has
     * to be refused, which is what says the first assertion is about the
     * rewrite being skipped rather than about the file being easy.
     * Input: five components tagged CMYK with a `cdef`, and five tagged sRGB
     * -> Output: `cmyk` from the first, `srgb` from the second, the same
     * samples from both, and a refusal from the first with its box removed.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn the_colr_box_only_goes_when_the_decoder_has_already_refused_the_file() {
        /// The same file with a `cdef` box spliced into its `jp2h`, marking
        /// the last channel an opacity and the rest colours.
        ///
        /// I.5.3.6: `N`, then `N` triples of `Cn`, `Typ`, `Asoc`. `Typ = 1`
        /// is opacity and `Asoc = 0` associates it with the whole image,
        /// which is how `jp2ksave` writes its own alpha (measured).
        fn with_cdef(bytes: &[u8], channels: u16) -> Vec<u8> {
            let mut payload = channels.to_be_bytes().to_vec();
            for channel in 0..channels {
                payload.extend_from_slice(&channel.to_be_bytes());
                let (kind, association) = if channel + 1 == channels {
                    (1u16, 0u16)
                } else {
                    (0, channel + 1)
                };
                payload.extend_from_slice(&kind.to_be_bytes());
                payload.extend_from_slice(&association.to_be_bytes());
            }
            let mut cdef = u32::try_from(payload.len() + 8)
                .expect("a cdef box is small")
                .to_be_bytes()
                .to_vec();
            cdef.extend_from_slice(b"cdef");
            cdef.extend_from_slice(&payload);

            let boxes = walk_boxes(bytes, 0, bytes.len()).expect("the encoder's container");
            let header = boxes
                .iter()
                .find(|b| &b.kind == b"jp2h")
                .expect("a jp2h box");
            let grown = u32::try_from(header.end - header.at + cdef.len()).expect("still small");
            let mut out = bytes[..header.at].to_vec();
            out.extend_from_slice(&grown.to_be_bytes());
            out.extend_from_slice(&bytes[header.at + 4..header.end]);
            out.extend_from_slice(&cdef);
            out.extend_from_slice(&bytes[header.end..]);
            out
        }

        let five = ramp(
            9,
            7,
            PixelFormat::Multi8(std::num::NonZeroU16::new(5).expect("5 is non-zero")),
        )
        .encode_jp2k(SaveOptions::default())
        .expect("five bands encode");

        let named = with_cdef(&retagged(&five, ENUMCS_CMYK), 5);
        let kept = decode_jp2k(&named, DecodeLimits::default())
            .expect("CMYK plus an opacity is five channels the decoder can name");
        assert_eq!(kept.format().channels(), 5);
        assert_eq!(kept.interpretation(), Interpretation::Cmyk);

        let rewritten = decode_jp2k(&retagged(&five, ENUMCS_SRGB), DecodeLimits::default())
            .expect("sRGB over five components is the refusal the removal answers");
        assert_eq!(rewritten.format().channels(), 5);
        assert_eq!(rewritten.interpretation(), Interpretation::Srgb);
        assert_eq!(
            rewritten.data(),
            kept.data(),
            "the same codestream under two boxes: only the interpretation may differ"
        );

        // The positive control. Both assertions above would pass against a
        // rewrite that never runs at all, so this is the one that says the
        // first file survived *because* the rewrite was skipped.
        let layout = ContainerLayout::parse(&named).expect("the spliced container");
        let stripped = layout
            .unspecified_rewrite(&named)
            .expect("a JP2 with a colr box has a removal");
        assert!(
            hayro_jpeg2000::Image::new(&stripped, &hayro_jpeg2000::DecodeSettings::default())
                .is_err(),
            "removing the box from a file the decoder can already name breaks it, which \
             is why the retry waits for a refusal"
        );
    }

    /**
     * Pins the two rewrites `ContainerLayout::unspecified_rewrite` makes,
     * directly, because the caller reaches it only after the decoder has
     * refused a file and end-to-end coverage of a caller says nothing about
     * the arms the caller cannot reach (issues #689, #882, #769).
     * The JP2 arm must produce a file whose `colr` box is gone and whose
     * `jp2h` still parses, which is the pair a length left unshrunk would
     * break. The bare arm must produce a container whose codestream is the
     * one that went in, byte for byte.
     * Input: a five-band JP2 and its bare codestream -> Output: a JP2 with
     * no enumerated colour space, and a wrapped codestream that still parses.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn the_unspecified_rewrite_removes_a_box_and_wraps_a_bare_codestream() {
        let five = ramp(
            9,
            7,
            PixelFormat::Multi8(std::num::NonZeroU16::new(5).expect("5 is non-zero")),
        )
        .encode_jp2k(SaveOptions::default())
        .expect("five bands encode");
        let layout = ContainerLayout::parse(&five).expect("the encoder's own container");
        assert_eq!(
            layout.enum_cs,
            Some(ENUMCS_SRGB),
            "the box this starts with"
        );

        let stripped = layout
            .unspecified_rewrite(&five)
            .expect("a JP2 with a colr box has a removal");
        assert_eq!(
            stripped.len(),
            five.len() - 15,
            "a METH=1 colr box is 15 bytes"
        );
        let after = ContainerLayout::parse(&stripped).expect("the shrunk jp2h still parses");
        assert_eq!(
            after.enum_cs, None,
            "with the box gone the decoder reaches its unspecified arm"
        );
        assert_eq!(
            &stripped[after.codestream..],
            &five[layout.codestream..],
            "the codestream is untouched; only the boxes above it moved"
        );

        // The bare arm, on the same codestream.
        let bare = five[layout.codestream..].to_vec();
        let bare_layout = ContainerLayout::parse(&bare).expect("a bare codestream");
        assert!(bare_layout.bare);
        let wrapped = bare_layout
            .unspecified_rewrite(&bare)
            .expect("a bare codestream is wrapped rather than edited");
        let wrapped_layout = ContainerLayout::parse(&wrapped).expect("the wrapper parses");
        assert_eq!(wrapped_layout.enum_cs, None);
        assert_eq!(
            &wrapped[wrapped_layout.codestream..],
            &bare[..],
            "the wrapper carries the codestream that went in"
        );
        // The positive control: the wrapper is not merely parseable, it is
        // the thing that makes the file readable at all.
        assert!(
            decode_jp2k(&bare, DecodeLimits::default()).is_ok(),
            "a bare five-component codestream decodes through the wrap"
        );
    }

    /**
     * Pins the tiled encoder against the oracle by whole-file digest: the
     * bytes this module writes for a tiled save are the bytes
     * `vips jp2ksave` writes for the same image and the same grid, all of
     * them (issue #768).
     * This is the assertion that says the splice is a rearrangement rather
     * than a re-encode. `Encoder::encode` writes one tile and there is no
     * knob to make it write more, because `openjpeg2-pure-rs` keeps its whole
     * `openjpeg` module `pub(crate)`; what makes a tiled file reachable
     * anyway is that JPEG 2000 codes every tile independently, so the
     * tile-part for a region is the same bytes whether it came from a tiled
     * encode or from a standalone one placed at the same absolute
     * coordinates. Byte identity with OpenJPEG's own tiled encoder is how
     * that stops being an argument.
     * Each row was captured as, for the 37x21 row:
     *
     * ```text
     * vips rawload ramp.raw ramp.v 37 21 3 --format uchar
     * vips copy ramp.v ramp.i.v --interpretation srgb
     * vips jp2ksave ramp.i.v out.jp2 --lossless --tile-width 16 --tile-height 16
     * ```
     *
     * where `ramp.raw` is this module's own `ramp` fixture, whose digest is
     * pinned beside the output's so a changed fixture is a failure here
     * rather than a comparison against a different image.
     * The untiled row is not filler: it pins that the default writes
     * `XTsiz = YTsiz = 512` into `SIZ` the way vips does, which is the one
     * byte a single-tile file used to differ by.
     * Input: three rasters at three grids -> Output: the exact files vips
     * writes.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn the_files_this_encoder_writes_are_the_files_vips_writes() {
        type Case = (
            u32,
            u32,
            PixelFormat,
            Option<Interpretation>,
            u32,
            u32,
            &'static str,
            &'static str,
        );
        let cases: [Case; 9] = [
            (
                37,
                21,
                PixelFormat::Rgb8,
                None,
                16,
                16,
                "7bf37b54668d747667f0ca67412f795f153c5b6d9523469a23fc0d04095cac1e",
                "e52bee5ba328433f3fb803bbf31575adf5b52fa8a1436076274e91135d2807e2",
            ),
            (
                37,
                21,
                PixelFormat::Rgb8,
                None,
                512,
                512,
                "7bf37b54668d747667f0ca67412f795f153c5b6d9523469a23fc0d04095cac1e",
                "07c7807a8cf432e8a8ff53a54387e817664ea7b3f77ca1b2d9ef8a349df22474",
            ),
            (
                70,
                50,
                PixelFormat::Gray16,
                None,
                32,
                32,
                "fe7a6b815be477ac34a943efcceacaa78650690be2815c46e600e2251f6e7add",
                "9073c8836c7a889d7dd0781ceb312305470406164fdddb6e62498284a4f570f0",
            ),
            // The row that needs more than one resolution level, and the only
            // one here that does. `num_resolutions` is
            // `max(1, floor(log2(min(w, h))) - 5)`, so it is 1 for every shape
            // above and 2 for this one, which is what makes the wavelet run at
            // all and the tile's absolute placement decide the coded bytes. A
            // 4x3 grid with partial tiles on both edges.
            (
                200,
                150,
                PixelFormat::Rgb8,
                None,
                64,
                64,
                "4ed24ad8b60b2a0119c3fc5b5bfc1b8c716ca92cd92044549a7ab9ba8d7a5b07",
                "9126c417a44238f3540e6735efea0f22741d07a6da32806b467dfb0b57cc7bdb",
            ),
            // The row where the tile's own coordinates decide the bytes, and
            // the reason it is 37x27 rather than another power of two. A tile
            // grid whose step is even and code-block aligned puts every tile
            // origin at the same parity and the same code-block offset as the
            // image origin, so encoding a tile as though it sat at (0, 0)
            // produces the same bytes and nothing catches it: measured, that
            // mutation is green against all four rows above. An odd step is
            // what separates them, because the wavelet's interleave parity is
            // `tcx0 % 2`.
            (
                200,
                150,
                PixelFormat::Rgb8,
                None,
                37,
                27,
                "4ed24ad8b60b2a0119c3fc5b5bfc1b8c716ca92cd92044549a7ab9ba8d7a5b07",
                "ea70e0999c28dc436d00fcb19b859a3fb35207cec738d8277f1227f07d978aaf",
            ),
            // The four rows that carry a `cdef` box, which is the whole of
            // issue #935: RGB plus alpha and greyscale plus alpha, at
            // both element widths, tiled and not.
            (
                24,
                18,
                PixelFormat::Rgba8,
                None,
                8,
                8,
                "a31a0a167e6a59c89e45dd81711e701b4c1bc95fe950fda92e1f3b87f4756eab",
                "fb20de0109771406dd8c73a51bf38bd18a3b053b7c5815e7ee5c36a5979cde3c",
            ),
            (
                20,
                14,
                PixelFormat::Rgba16,
                None,
                512,
                512,
                "1c8be83b916938b338b04a95b4e941f315886af2082461222110572ec4c84a0a",
                "a02e3d5b4226f95177f29e583ae7c2e07255bb58c839e860b75366be70e9abdc",
            ),
            (
                30,
                20,
                PixelFormat::Multi8(std::num::NonZeroU16::new(2).expect("2 is non-zero")),
                Some(Interpretation::Bw),
                16,
                16,
                "fe6c69f1360e8c8725a77e70f02ef08bfe95e98061bfc70d90eea0bd3d141a9b",
                "1ad7945dd499ebba063510f11b4c87a9d0b79df309c172bf57b9cc3ebcecf2fb",
            ),
            (
                30,
                20,
                PixelFormat::Multi16(std::num::NonZeroU16::new(2).expect("2 is non-zero")),
                Some(Interpretation::Grey16),
                512,
                512,
                "b7ec85a75c62603248a9a1d9cbab3dfd61237b620e5efc6772fb7541ec437a20",
                "2869b5d496697f95fdf807790104c55542cea5c593015ac7e50582ed27b49f1a",
            ),
        ];
        for (width, height, format, interpretation, tile_w, tile_h, fixture, expected) in cases {
            let source = match interpretation {
                // vips writes the `cdef` box off the interpretation and not off
                // the band count, so a two-band raster has to be tagged for it
                // to be greyscale plus alpha rather than a compute
                // intermediate. `Rgba8` and `Rgba16` need no tag: they are
                // `srgb` and `rgb16` already.
                Some(tag) => ramp(width, height, format)
                    .copy()
                    .interpretation(tag)
                    .build(),
                None => ramp(width, height, format),
            };
            assert_eq!(
                payload_digest(&source),
                fixture,
                "{width}x{height} {format:?}: the fixture the oracle saw, so a changed \
                 ramp fails here rather than comparing a different image"
            );
            let bytes = source
                .encode_jp2k(
                    SaveOptions::default()
                        .with_tile_width(NonZeroU32::new(tile_w).expect("non-zero"))
                        .with_tile_height(NonZeroU32::new(tile_h).expect("non-zero")),
                )
                .expect("a tiled encode");
            assert_eq!(
                digest(&bytes),
                expected,
                "{width}x{height} {format:?} at {tile_w}x{tile_h}: {} bytes that are not \
                 the ones vips writes",
                bytes.len()
            );
        }
    }

    /**
     * Pins the `cdef` box against the whole of vips's rule rather than
     * against the two shapes that get one, because a rule this narrow is easy
     * to write too widely (issue #935).
     * `jp2ksave` writes a channel definition box for greyscale plus one band
     * and for RGB plus one band and for nothing else. Measured over six band
     * counts and six interpretations, 36 files: two carry a box. It is **not**
     * `vips_image_hasalpha`, which is true for two bands under any tag and for
     * anything past four, so CMYK plus one and a five-band image are the rows
     * that separate the real rule from the obvious one, and both are here.
     * The untagged two-band raster is the third: this module writes it a
     * greyscale `colr` box, so a rule keyed on that box rather than on the
     * interpretation would give a compute intermediate an alpha channel.
     * The positive control is the last row, which must carry the box with the
     * exact entries, or the whole sweep passes by writing none anywhere.
     * Input: six shapes -> Output: a box on one of them and its four entries.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn the_channel_definition_box_goes_on_the_two_shapes_vips_puts_it_on() {
        /// The `cdef` box's entries, or `None` when the file carries no such
        /// box.
        fn entries(bytes: &[u8]) -> Option<Vec<(u16, u16, u16)>> {
            let top = walk_boxes(bytes, 0, bytes.len()).expect("the encoder's container");
            let header = top.iter().find(|b| &b.kind == b"jp2h")?;
            let children = walk_boxes(bytes, header.start, header.end).expect("jp2h children");
            let cdef = children.iter().find(|b| &b.kind == b"cdef")?;
            let payload = &bytes[cdef.start..cdef.end];
            let count = usize::from(u16::from_be_bytes([payload[0], payload[1]]));
            assert_eq!(
                payload.len(),
                2 + count * 6,
                "a cdef box is N then N triples, so a length that does not fit means the \
                 splice wrote a box nobody can read"
            );
            Some(
                (0..count)
                    .map(|i| {
                        let at = 2 + i * 6;
                        let field =
                            |o: usize| u16::from_be_bytes([payload[at + o], payload[at + o + 1]]);
                        (field(0), field(2), field(4))
                    })
                    .collect(),
            )
        }

        let two = std::num::NonZeroU16::new(2).expect("2 is non-zero");
        let five = std::num::NonZeroU16::new(5).expect("5 is non-zero");
        let none: [(PixelFormat, Option<Interpretation>, &str); 5] = [
            (
                PixelFormat::Gray8,
                None,
                "one band is a colour channel and no more",
            ),
            (PixelFormat::Rgb8, None, "three bands fill sRGB exactly"),
            (
                PixelFormat::Rgba8,
                Some(Interpretation::Cmyk),
                "CMYK plus one gets no box, which is where hasalpha and vips disagree",
            ),
            (
                PixelFormat::Multi8(five),
                None,
                "five bands get no box either, the second place hasalpha would",
            ),
            (
                PixelFormat::Multi8(two),
                None,
                "an untagged two-band raster is a compute intermediate, not grey plus alpha",
            ),
        ];
        for (format, tag, why) in none {
            let source = match tag {
                Some(tag) => ramp(9, 7, format).copy().interpretation(tag).build(),
                None => ramp(9, 7, format),
            };
            let bytes = source
                .encode_jp2k(SaveOptions::default())
                .unwrap_or_else(|e| panic!("{format:?}: {e}"));
            assert_eq!(entries(&bytes), None, "{format:?}: {why}");
        }

        // The positive control, without which every row above passes against
        // an encoder that writes no box at all.
        let rgba = ramp(9, 7, PixelFormat::Rgba8)
            .encode_jp2k(SaveOptions::default())
            .expect("RGBA encodes");
        assert_eq!(
            entries(&rgba),
            Some(vec![(0, 0, 1), (1, 0, 2), (2, 0, 3), (3, 1, 0)]),
            "the entries vips writes: each colour channel typed 0 with a one-based \
             association, and the alpha typed 1 against the whole image"
        );
        let grey = ramp(9, 7, PixelFormat::Multi8(two))
            .copy()
            .interpretation(Interpretation::Bw)
            .build()
            .encode_jp2k(SaveOptions::default())
            .expect("grey plus alpha encodes");
        assert_eq!(entries(&grey), Some(vec![(0, 0, 1), (1, 1, 0)]));
    }

    /**
     * Pins the tiled save end to end: the grid reaches the file, the file
     * reads back, and the pixels are the untiled ones (issue #768).
     * Three assertions, and the middle one is the reason the first is not
     * enough. A `tile-width` of 16 in the metadata says the `SIZ` marker says
     * 16; it does not say the tile-parts after it are the right bytes in the
     * right order. Comparing the decode against the untiled save's decode is
     * what says that, and comparing both against the source is what says
     * neither is wrong in the same way.
     * The 37x21 shape is deliberate: at 16x16 it is a 3x2 grid with partial
     * tiles down two edges, so a splice that assumed full tiles fails here.
     * Input: a 37x21 RGB ramp at 16x16 and at the default 512x512 ->
     * Output: `tile-width` / `tile-height` of 16 on the first and neither on
     * the second, with identical pixels from both.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn a_tiled_save_round_trips_and_reports_the_grid_it_was_given() {
        let source = ramp(37, 21, PixelFormat::Rgb8);
        let sixteen = NonZeroU32::new(16).expect("16 is non-zero");
        let tiled = source
            .encode_jp2k(
                SaveOptions::default()
                    .with_tile_width(sixteen)
                    .with_tile_height(sixteen),
            )
            .expect("a tiled encode");
        let untiled = source
            .encode_jp2k(SaveOptions::default())
            .expect("the default encode");

        // A grid whose step is odd, so the tile origins do not share the image
        // origin's parity. This is the shape that needs the tile's absolute
        // coordinates to reach the encoder: on an even, code-block-aligned
        // step every tile codes the same either way.
        let odd = ramp(200, 150, PixelFormat::Rgb8);
        let odd_tiled = odd
            .encode_jp2k(
                SaveOptions::default()
                    .with_tile_width(NonZeroU32::new(37).expect("37 is non-zero"))
                    .with_tile_height(NonZeroU32::new(27).expect("27 is non-zero")),
            )
            .expect("an odd tiled encode");
        let from_odd = decode_jp2k(&odd_tiled, DecodeLimits::default()).expect("odd decode");
        assert_eq!(int_field(&from_odd, "tile-width"), Some(37));
        assert_eq!(
            from_odd.data(),
            odd.data(),
            "a 6x6 grid on an odd step has to be lossless too"
        );

        let from_tiled = decode_jp2k(&tiled, DecodeLimits::default()).expect("tiled decode");
        let from_untiled = decode_jp2k(&untiled, DecodeLimits::default()).expect("untiled decode");

        assert_eq!(int_field(&from_tiled, "tile-width"), Some(16));
        assert_eq!(int_field(&from_tiled, "tile-height"), Some(16));
        assert_eq!(
            int_field(&from_untiled, "tile-width"),
            None,
            "a 37x21 image at 512x512 is one tile, and vips attaches nothing there"
        );

        assert_eq!(
            from_tiled.data(),
            source.data(),
            "a 3x2 grid with partial tiles on two edges has to be lossless"
        );
        assert_eq!(
            from_tiled.data(),
            from_untiled.data(),
            "the tiling is a container decision and must not move a sample"
        );
    }

    /**
     * Pins that the default tile size is `jp2ksave`'s own 512 rather than
     * "the whole image", which is the half of #768 the issue does not
     * mention: vips tiles by default, so an encoder that always writes one
     * tile diverges on every image larger than 512.
     * Measured: `vips jp2ksave` on a 600x600 image with no tile options
     * writes a file `vipsheader` reports as `tile-width: 512`.
     * The 500x500 row is the control that says the constant is a tile size
     * and not a flag: the same default writes one tile there, and one tile
     * carries no geometry at all.
     * Input: 600x600 and 500x500 at `SaveOptions::default` -> Output: a
     * 512 grid on the first and none on the second.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn the_default_tile_size_is_the_one_vips_defaults_to() {
        assert_eq!(SaveOptions::default().tile_width, DEFAULT_TILE_SIZE);
        assert_eq!(SaveOptions::default().tile_height, DEFAULT_TILE_SIZE);
        assert_eq!(DEFAULT_TILE_SIZE.get(), 512);

        let big = ramp(600, 600, PixelFormat::Gray8)
            .encode_jp2k(SaveOptions::default())
            .expect("a 600x600 default save");
        let big = decode_jp2k(&big, DecodeLimits::default()).expect("decode");
        assert_eq!(int_field(&big, "tile-width"), Some(512));
        assert_eq!(int_field(&big, "tile-height"), Some(512));

        let small = ramp(500, 500, PixelFormat::Gray8)
            .encode_jp2k(SaveOptions::default())
            .expect("a 500x500 default save");
        let small = decode_jp2k(&small, DecodeLimits::default()).expect("decode");
        assert_eq!(int_field(&small, "tile-width"), None);
    }

    /**
     * Pins the tile-count ceiling as a typed refusal naming the same limit
     * OpenJPEG names, because `Isot` is two bytes and a grid past it would
     * otherwise be spliced with wrapped tile indices (issue #768).
     * Measured on the oracle: `vips jp2ksave` on a 256x256 image at one pixel
     * per tile fails with "Invalid number of tiles : 256 x 256 (maximum fixed
     * by jpeg2000 norm is 65535 tiles)", and 255x255 gets past that check and
     * fails on the resolution count instead.
     * The refusal is before any encode, which is what the control says: a
     * 255x255 grid at the same tile size gets far enough to be refused by the
     * encoder rather than by this check, and the two messages are different.
     * Input: 256x256 and 255x255 at one pixel per tile -> Output: this
     * module's refusal for the first and the encoder's for the second.
     */
    #[test]
    #[cfg(feature = "jp2k")]
    fn a_tile_grid_past_isot_is_refused_before_anything_is_encoded() {
        let one = NonZeroU32::new(1).expect("1 is non-zero");
        let over = Raster::zeroed(256, 256, PixelFormat::Gray8)
            .expect("a raster")
            .encode_jp2k(
                SaveOptions::default()
                    .with_tile_width(one)
                    .with_tile_height(one),
            )
            .expect_err("65536 tiles is one past Isot");
        let message = over.to_string();
        assert!(
            message.contains("65535") && message.contains("65536"),
            "the refusal names the limit and the count: {message}"
        );

        // The control: one tile fewer is refused by the encoder instead, so
        // the check above is a count and not a blanket refusal of small
        // tiles.
        let under = Raster::zeroed(255, 255, PixelFormat::Gray8)
            .expect("a raster")
            .encode_jp2k(
                SaveOptions::default()
                    .with_tile_width(one)
                    .with_tile_height(one),
            )
            .expect_err("a one-pixel tile cannot carry the resolution count");
        assert!(
            !under.to_string().contains("65535"),
            "65025 tiles is inside Isot, so this must be the encoder's refusal and not \
             this module's: {under}"
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
     * turns the transform on. libviprs used to match the first two and
     * **refuse** the third, because `hayro-jpeg2000` will not parse a `colr`
     * box it does not recognise; #771 fixed that by keeping the enum away from
     * the decoder, and the third row is now the one that proves the
     * replacement rule works, since the transform it needs can only come from
     * `SIZ`.
     * Input: three hand-wrapped JP2s -> Output: three decodes, each matching
     * vips.
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

        // EnumCS 99, not a defined value: UNSPECIFIED, so the subsampling in
        // `SIZ` is the only thing that can turn the transform on, and it has
        // to (issue #771). This is the row the decoder used to refuse outright,
        // and it is the one that says the replacement rule is doing the work
        // rather than the enum having been quietly honoured: 99 names nothing,
        // so the only route to `[255, 87, 0]` is through `SIZ`.
        let unknown = decode_jp2k(&wrap(99), DecodeLimits::default())
            .expect("an unrecognised colr box must not stop the codestream decoding");
        let got = &samples(&unknown)[..3];
        for (band, (mine, want)) in got.iter().zip([255u32, 87, 0].iter()).enumerate() {
            assert!(
                mine.abs_diff(*want) <= 1,
                "band {band}: {mine} against vips's {want}"
            );
        }

        // And the control that keeps that from being "everything gets the
        // transform": the sRGB row above is the same codestream, subsampled in
        // exactly the same way, and must stay untransformed.
        assert_eq!(&samples(&srgb)[..3], &[128, 16, 240]);
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
        fn palettised(depth: u8, enumcs: u32) -> Vec<u8> {
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
            colr.extend_from_slice(&enumcs.to_be_bytes());

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
        let bytes = palettised(8, 16);
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

        // The same palette under a `colr` box nobody recognises, which is the
        // case that says why #771's rewrite neutralises to **sRGB** and not to
        // the greyscale a one-component `SIZ` would suggest. The palette turns
        // one component into three, so a greyscale neutral would hand the
        // decoder a colour space with one channel and the decode would come
        // back `BandCountMismatch`.
        //
        // vips is broken here, measured on 8.18.6: it reports
        // `5x1 uchar, 1 band, b-w` and then any pixel read fails with
        // "decoded image does not match container", because it guesses the
        // interpretation from the component count **before** the palette. This
        // keeps the three real bands and reads them, which is the same call
        // #767 made for the one-component sRGB file.
        let unknown_enum = decode_jp2k(&palettised(8, 99), DecodeLimits::default())
            .expect("an unrecognised colr box must not cost a palettised file its palette");
        assert_eq!(unknown_enum.format(), PixelFormat::Rgb8);
        assert_eq!(
            samples(&unknown_enum)[..9],
            samples(&raster)[..9],
            "the colr box does not touch the palette, so the samples are the same nine"
        );
        assert_eq!(
            unknown_enum.interpretation(),
            Interpretation::Srgb,
            "unrecognised, so the band count decides, and it is three after the palette"
        );

        // The 16-bit palette: wider than the index SIZ declared, so the
        // carrier the frame was priced for cannot hold it.
        let err = decode_jp2k(&palettised(16, 16), DecodeLimits::default())
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
