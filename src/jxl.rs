//! Still-image JPEG XL (`.jxl`) load and lossless save: every codestream
//! this build can meet in, a lossless modular one out.
//!
//! Ported from libvips `foreign/jxlload.c` and `foreign/jxlsave.c` (v8.18.4
//! for the line numbers quoted below; every measured number comes from the
//! Homebrew 8.18.4 binary, which is a different artefact). libvips wraps
//! libjxl and so reaches the whole format in both directions; libviprs
//! reaches it through two crates that split the job unevenly, `jxl-oxide`
//! for the decode and `zune-jpegxl` for the encode. That asymmetry is the
//! shape of this module and the reason [`SaveOptions`] looks nothing like
//! `jxlsave`'s ten options.
//!
//! Both crates and their trees sit behind the non-default **`jxl`**
//! feature, the way `resvg` sits behind `svg` and for the same reason:
//! they cost more lock entries than the rest of the codec surface put
//! together, and one of them is `tracing`, which this crate otherwise
//! keeps opt-in. The `Cargo.toml` comment on the feature carries the
//! measured numbers. Without it every entry point below still exists,
//! still compiles and still has the same signature; each one returns a
//! typed refusal naming the feature instead of doing the work, so a
//! caller's code does not change shape with the build. [`JxlError`] and
//! the [`SourceError::Jxl`] variant carrying it are declared in both
//! builds for the same reason, and
//! [`JxlError::FeatureNotEnabled`] is the arm the feature-off build takes.
//!
//! # Operations
//!
//! | libviprs method | libvips equivalent | result |
//! |---|---|---|
//! | [`decode_jxl`] | `jxlload` / `jxlload_buffer` (default `n = 1`) | 8-bit, 16-bit or float raster, plus `icc-profile-data` / `exif-data` / `xmp-data` / `bits-per-sample` / `n-pages` |
//! | [`Raster::encode_jxl`] | `jxlsave_buffer --lossless --keep none` | `.jxl` bytes |
//! | [`Raster::save_jxl`] | `jxlsave --lossless --keep none` | `.jxl` file |
//!
//! # Semantics
//!
//! * **Decode is close to exact, and the two paths differ in how close.**
//!   libjxl and `jxl-oxide` both target the JPEG XL conformance suite, so
//!   this is a parity port rather than an approximation. Measured on the
//!   captures in `oracle-captures/foreign-jxl/`: the **lossless modular**
//!   path is a true identity for all three carriers, 8-bit, 16-bit and
//!   float, so its pins carry no tolerance at all. The **VarDCT** path (a
//!   default `jxlsave`, which targets a butteraugli distance) agrees with
//!   libjxl to within **one count per channel** and is pinned with exactly
//!   that tolerance and no more. If a lossless round trip ever stops being
//!   bit-exact, something is wrong rather than merely imprecise.
//! * **The encoder is lossless and nothing else.** `zune-jpegxl` 0.5.2 is a
//!   lossless modular encoder: `encoder.rs` has no VarDCT path, no
//!   butteraugli target and no quantisation step anywhere in it. That is
//!   why [`SaveOptions`] carries a [`Compression`] rather than a
//!   `distance: f64` or a `quality: u8`. An argument the encoder throws
//!   away inverts the contract (ask for distance 3, get a lossless file
//!   several times the size you asked for) and is a semver time bomb,
//!   because the day a VarDCT encoder lands every existing
//!   `encode_jxl(3.0)` would silently start emitting lossy files in a patch
//!   release. Making lossy unrepresentable turns that into a compile error
//!   today instead, and [`Compression`] is `#[non_exhaustive]` so
//!   `Lossy { .. }` can join it as a minor bump.
//! * **Not implemented, and not silently accepted either.** Of `jxlsave`'s
//!   options, libviprs honours none of them, and none of them is a field
//!   this module ignores: `distance`, `Q`, `lossless = false`, `tier`,
//!   `effort`, `bitdepth`, `keep`, `background`, `page-height` and
//!   `profile` have no spelling in [`SaveOptions`] at all. `zune-jpegxl`
//!   does expose an `effort` knob on its `EncoderOptions`, and it is
//!   deliberately left at the crate default: it is not `jxlsave`'s speed
//!   tier but a sampling radius for the prefix-code histograms
//!   (`encoder.rs:1089` and `:1138`), so it moves the compressed size and never
//!   the decoded pixels. Mapping `--effort 7` onto it would be a lie about
//!   what the number means.
//! * **16-bit is written, not refused.** This is where JPEG XL and WebP
//!   part company. WebP has no 16-bit sample so
//!   [`Raster::encode_webp`](crate::Raster::encode_webp) refuses a wide
//!   raster rather than narrowing it; JPEG XL holds 16-bit natively and so
//!   does `zune-jpegxl`, so there is no narrowing question here and
//!   [`PixelFormat::Gray16`], [`PixelFormat::Rgb16`] and
//!   [`PixelFormat::Rgba16`] all encode exactly.
//! * **Greyscale stays one band, also unlike WebP.** `webpsave` promotes a
//!   `b-w` image to three bands because the format stores no greyscale;
//!   `jxlsave` does not, because it does. Measured: a 4x3 `b-w` uchar
//!   image round-trips as `1 band, b-w`.
//! * **Two pixels is the floor on each axis.** `vips jxlsave` accepts every
//!   geometry down to 1x1 (measured: 1x1, 2x1, 1x2 and 4x1 all write and
//!   read back). `zune-jpegxl`'s `encoder.rs:1059-1064` rejects `width <= 1`
//!   or `height <= 1` outright, so libviprs has a floor vips does not and
//!   it is [`MIN_DIMENSION`]. The refusal names the floor rather than
//!   letting the dependency's `ZeroDimension("width")` reach the caller.
//! * **An animation loads frame 0 and says so.** A default `vips jxlload`
//!   reads one frame and sets `n-pages` to the count in the original
//!   (`jxlload.c:743-751`); the 4x9 toilet roll needs `[n=-1]`.
//!   [`decode_jxl`] matches that default exactly. Reading every frame is
//!   issue #621 and waits on the page model in #564.
//! * **`icc-profile-data` is the profile the pixels are in, and is always
//!   present.** `jxlload.c:955-985` asks libjxl for
//!   `JXL_COLOR_PROFILE_TARGET_DATA`, which synthesises a profile when the
//!   file embeds none, so a vips-loaded JXL always carries one.
//!   [`decode_jxl`] uses `jxl_oxide::JxlImage::rendered_icc`, which has the
//!   same contract: an embedded profile comes back verbatim, and an enum
//!   colour encoding is turned into a profile describing it. The two
//!   generators do not agree byte for byte, and cannot: measured on the
//!   same sRGB file, libjxl writes 504 bytes and `jxl-color` writes 572.
//!   The profiles describe the same space; only the bytes differ.
//! * **The EXIF box is not the EXIF blob.** JPEG XL stores the TIFF block
//!   behind a big-endian 4-byte offset and *without* the `Exif\0\0` prefix
//!   a JPEG APP1 segment carries, so `jxlload.c:630-664` skips the offset
//!   and glues the prefix back on. [`decode_jxl`] does the same, which is
//!   what makes a JXL `exif-data` blob compare equal to the JPEG one for
//!   the same image (measured: a JPEG transcoded to JXL with `--keep all`
//!   comes back with the identical 186-byte blob).
//! * **A malformed EXIF box costs the blob, not the image.** vips fails the
//!   whole load when the offset runs past the payload, warning `invalid
//!   data in EXIF box` (`jxlload.c:646-649`, measured: `vipsheader` exits
//!   1 and prints no header). libviprs deliberately diverges and drops the
//!   blob instead, because refusing an otherwise-valid image over a
//!   metadata box is the wrong trade for a decoder that eats untrusted
//!   bytes. The same divergence runs the other way for the nonconforming
//!   box whose payload already carries the `Exif\0\0` prefix: vips
//!   special-cases it (`jxlload.c:635-637`) where `jxl-oxide` reads the
//!   first four bytes as an offset regardless, so libviprs drops a blob
//!   vips keeps.
//! * **The orientation is applied, not tagged.** `jxlload.c:822` copies the
//!   header's orientation into the `orientation` field and leaves the
//!   pixels alone, so `vips_autorot` is what finally turns them. Every
//!   pixel accessor `jxl-oxide` exposes applies the orientation itself, so
//!   [`decode_jxl`] hands back upright pixels and an `orientation` of 1,
//!   which is where `autorot` would have left them. The divergence is only
//!   reachable with a third-party file: `jxlsave.c` never sets the field,
//!   so vips itself writes nothing but 1.
//! * **Save writes a bare codestream and no metadata.** `zune-jpegxl`
//!   emits the codestream with no ISOBMFF container around it, and JPEG XL
//!   has nowhere else to put an EXIF block or an XMP packet (the ICC
//!   profile lives inside the codestream, which is further out of reach
//!   again). `vips jxlsave --keep none` writes the same bare-codestream
//!   form, so the shapes agree where they overlap and `--keep all` simply
//!   has no encoder behind it here. [`SaveOptions`] therefore has no
//!   `keep` field, and [`Raster::save`](crate::Raster::save) and
//!   [`Raster::save_stripped`](crate::Raster::save_stripped) write
//!   identical `.jxl` bytes.
//! * **CMYK is refused rather than mislabelled.** libjxl counts a JPEG XL
//!   `Black` channel as an extra channel, so `jxlload.c:698-737` switches
//!   on three colour channels and tags a CMYK file `srgb` with four bands,
//!   which is not what those four bands are. `jxl-oxide` reports the
//!   colour space honestly, so [`decode_jxl`] can see the black channel
//!   and refuses by name. Neither of the two ways out is wired. Converting
//!   the inks means an ICC transform through the file's own profile, and
//!   `jxl-oxide` only runs one when a `ColorManagementSystem` has been
//!   handed to `JxlImage::set_cms`; this build hands it none, so the
//!   default `jxl_color::NullCms` refuses every transform
//!   (`jxl-color-0.11.0/src/cms.rs:47-57`, reached through
//!   `jxl-render-0.12.4/src/lib.rs:184`), and jxl-oxide's own `lcms2` and
//!   `moxcms` integrations are optional and both off here.
//!   [`JxlError::CmykNotSupported`] is the refusal. Carrying the
//!   inks through untouched means a CMYK route into
//!   [`crate::colour`], which does hold a black channel
//!   ([`Interpretation::Cmyk`](crate::conversion::Interpretation::Cmyk),
//!   the naive ink model and profiled CMYK through
//!   [`Raster::icc_import`](crate::Raster::icc_import)) but has no edge
//!   from this loader. So the refusal is a wiring gap and not a
//!   capability the crate lacks. Nothing in this build can produce such a
//!   file to test it against either: `vips jxlsave` converts a `cmyk`
//!   image to sRGB on the way in.
//!
//! Every number this module is pinned against was measured on the real
//! vips 8.18.4 binary and is recorded, with the commands that produced it,
//! in `oracle-captures/foreign-jxl/`.
//!
//! Every entry point here is fallible and there is no panicking twin,
//! matching the rest of the codec surface in [`crate::encode`],
//! [`crate::radiance`], [`crate::gif`] and [`crate::webp`]: a decoder's
//! failures come from untrusted bytes, so a panicking spelling would have
//! no honest caller.
//!
//! The loader's refusals are [`JxlError`], reached through
//! [`SourceError::Jxl`], which is the shape
//! [`ExrError`](crate::exr::ExrError),
//! [`FitsError`](crate::fits::FitsError),
//! [`GifError`](crate::gif::GifError) and
//! [`RadianceError`](crate::radiance::RadianceError) already use. The
//! encoder's stay on the shared [`EncodeError`] spine, which is where
//! [`crate::gif`], [`crate::radiance`] and [`crate::fits`] leave their
//! save refusals too, so JPEG XL does not become the one codec here with
//! a third convention (issue #634).

use std::path::Path;

#[cfg(feature = "jxl")]
use std::num::NonZeroU16;

#[cfg(feature = "jxl")]
use jxl_oxide::{AllocTracker, AuxBoxData, InitializeResult, JxlImage};
use thiserror::Error;
#[cfg(feature = "jxl")]
use zune_core::bit_depth::BitDepth;
#[cfg(feature = "jxl")]
use zune_core::colorspace::ColorSpace;
#[cfg(feature = "jxl")]
use zune_core::options::EncoderOptions;
#[cfg(feature = "jxl")]
use zune_jpegxl::JxlSimpleEncoder;

use crate::codec::EncodeError;
#[cfg(feature = "jxl")]
use crate::conversion::Interpretation;
#[cfg(feature = "jxl")]
use crate::imageio::MetadataValue;
use crate::imageio::SaveError;
#[cfg(feature = "jxl")]
use crate::pixel::PixelFormat;
#[cfg(feature = "jxl")]
use crate::raster::decode_alloc_bytes;
use crate::raster::{Raster, RasterError};
use crate::source::{DecodeLimits, SourceError};

/// The smallest width or height [`Raster::encode_jxl`] will encode.
///
/// `zune-jpegxl` 0.5.2 rejects `width <= 1` and `height <= 1` before it
/// looks at anything else (`encoder.rs:1059-1064`), so a single-pixel row
/// or column has no encoder behind it. `vips jxlsave` has no such floor and
/// writes an 18-byte 1x1 file happily, which is why this constant is
/// documented rather than assumed: it is a libviprs limit, not a format
/// one.
pub const MIN_DIMENSION: u32 = 2;

/// The `Exif\0\0` prefix a JPEG APP1 segment carries and a JPEG XL `Exif`
/// box does not (`jxlload.c:657`).
///
/// Putting it back is what makes a JXL `exif-data` blob compare equal to
/// the JPEG one for the same image.
#[cfg(feature = "jxl")]
const EXIF_PREFIX: &[u8] = b"Exif\0\0";

/// Errors from the JPEG XL loader.
///
/// Every variant except [`JxlError::Raster`] and [`JxlError::Decode`]
/// describes a specific thing this build will not carry, which is what
/// makes them worth typing: the fuzz corpus in `fuzz/corpus/fuzz_jxl/`
/// asserts on the variant, not on a message.
///
/// The enum, and the [`SourceError::Jxl`] variant that carries it, are
/// declared whether or not the **`jxl`** feature is on, so a caller's
/// `match` has the same arms in either build and none of them names a type
/// that is not there. What changes is which arms are reachable: without the
/// feature the only one is [`JxlError::FeatureNotEnabled`], and with it
/// that is the only one that never fires.
///
/// The encoder does not report through here. [`Raster::encode_jxl`] and
/// [`Raster::save_jxl`] stay on the shared [`EncodeError`] spine, which is
/// where [`crate::gif`], [`crate::radiance`] and [`crate::fits`] leave
/// their save refusals too; this enum is the loader's, the way
/// [`ExrError`](crate::exr::ExrError) and
/// [`FitsError`](crate::fits::FitsError) are.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum JxlError {
    /// The crate was built without the **`jxl`** feature, so there is no
    /// decoder behind [`decode_jxl`] at all.
    ///
    /// Reported instead of a missing symbol or a panic: every entry point
    /// in this module exists at the same signature in both builds, so a
    /// caller's code does not change shape with the feature. This is the
    /// only variant a build without the feature can produce, and it is the
    /// one variant a build *with* it never produces, which is what makes
    /// "this build has no JPEG XL" distinguishable from "these bytes are
    /// not JPEG XL" without reading a message.
    #[error("jxl: JPEG XL decoding is not available in this build (enable the `jxl` feature)")]
    FeatureNotEnabled,
    /// `jxl-oxide` refused the bitstream: a bad signature, a malformed
    /// header, a broken entropy stream, an unreadable box.
    ///
    /// This is the untyped tail, for the reason [`ExrError::Decode`] is
    /// one: JPEG XL is a bitstream parser, an entropy coder, a modular
    /// predictor tree, a VarDCT inverse transform and an ISOBMFF box
    /// reader across thirteen crates, and reproducing that taxonomy here
    /// would be a second decoder. The message is `jxl-oxide`'s own, which
    /// carries its whole source chain inline (`jxl-render-0.12.4/src/error.rs:54-74`
    /// writes the inner error into every arm), so nothing is lost by
    /// flattening it to a string here.
    ///
    /// [`ExrError::Decode`]: crate::exr::ExrError::Decode
    #[error("jxl: {message}")]
    Decode {
        /// The underlying decoder failure, rendered through its `Display`.
        message: String,
    },
    /// The bytes ran out before the decoder had what it needed, named by
    /// what was still missing.
    ///
    /// Distinct from [`JxlError::Decode`] because the two-phase feed makes
    /// truncation its own answer rather than a parse failure: `jxl-oxide`
    /// says `NeedMoreData` instead of erroring, so a stream that simply
    /// stops is not malformed, it is short.
    #[error("jxl: the stream ended early, before {expected}")]
    Truncated {
        /// What the decoder was still waiting for.
        expected: &'static str,
    },
    /// The file carries a black ink channel, so it is CMYK rather than
    /// something this loader has a carrier and a tag for.
    ///
    /// Neither of the two ways out is wired. Converting the inks means an
    /// ICC transform, and `jxl-oxide` only runs one when a
    /// `ColorManagementSystem` has been handed to `JxlImage::set_cms` at
    /// runtime; this build hands it none and neither of the crate's own
    /// `lcms2` and `moxcms` integrations is enabled, so the default
    /// `jxl_color::NullCms` refuses every transform
    /// (`jxl-color-0.11.0/src/cms.rs:47-57`, installed at
    /// `jxl-render-0.12.4/src/lib.rs:184`). Carrying the inks through
    /// untouched means a CMYK route into [`crate::colour`], which does
    /// hold a black channel but has no edge from this loader. So the
    /// refusal is a wiring gap and not a capability the crate lacks.
    ///
    /// libjxl counts the `Black` channel as an extra channel instead, so
    /// `jxlload.c:698-737` tags such a file `srgb` with four bands, which
    /// is not what those four bands are.
    #[error(
        "jxl: this build cannot decode a {pixel_format} JPEG XL: converting the inks \
         needs an ICC transform and no ColorManagementSystem is installed on the \
         decoder, so jxl-oxide's default NullCms refuses one; carrying them through \
         untouched needs a CMYK route into `crate::colour`, which this loader does \
         not have yet"
    )]
    CmykNotSupported {
        /// The colour space `jxl-oxide` reports, `Cmyk` or `Cmyka`.
        pixel_format: String,
    },
    /// The declared channel count has no
    /// [`PixelFormat`](crate::pixel::PixelFormat) carrier.
    ///
    /// Defensive: `jxl_oxide::PixelFormat` names at most five channels
    /// today, so nothing `jxl-oxide` can report reaches this. It exists
    /// because the carrier is chosen from a number rather than from an
    /// enum, and a band count of zero or one above the
    /// [`FloatF32`](crate::pixel::PixelFormat::FloatF32) ceiling would
    /// otherwise be a panic or a zero-sized buffer.
    #[error(
        "jxl: a JPEG XL with {channels} channels has no raster carrier; a raster holds 1 to {max} bands"
    )]
    UnsupportedChannelCount {
        /// The channel count the header declared.
        channels: u32,
        /// The ceiling, `u16::MAX`.
        max: u32,
    },
    /// The rendered frame carries a different number of channels than the
    /// header declared.
    ///
    /// Defensive, and load-bearing: the header's count is what the
    /// allocation budget below was spent on and what the output buffer is
    /// sized for, so a frame that disagrees would be written into a buffer
    /// sized for something else.
    #[error(
        "jxl: the header declared {declared} channels and the rendered frame carries {rendered}"
    )]
    ChannelCountMismatch {
        /// The channel count the image header declared.
        declared: u32,
        /// The channel count the rendered frame produced.
        rendered: u32,
    },
    /// Decoding the declared geometry would allocate more than
    /// [`DecodeLimits::max_alloc_bytes`].
    ///
    /// Priced from the header alone, before a byte of frame data is fed
    /// in, which is the point of splitting the feed in two: a JPEG XL
    /// header declares its geometry in a handful of bytes and the body is
    /// entropy-coded, so the file is no guide to the frame's size. The
    /// pixel ceiling does not imply this one, since a pixel count sees
    /// neither the band count nor the sample depth: a 1-gigapixel
    /// `max_pixels` still permits an 8 GiB `Rgba16` frame.
    #[error(
        "jxl: decoding {width}x{height}x{channels} needs {needed} bytes, above the \
         {max_alloc_bytes}-byte decode allocation budget"
    )]
    AllocLimitExceeded {
        /// The declared frame width.
        width: u32,
        /// The declared frame height.
        height: u32,
        /// The declared channel count, colour channels plus alpha.
        channels: u32,
        /// Bytes the decoded raster would need.
        needed: u64,
        /// The budget in force, [`DecodeLimits::max_alloc_bytes`].
        max_alloc_bytes: u64,
    },
    /// `jxl-oxide`'s own allocation tracker refused a request from inside
    /// the decoder.
    ///
    /// Separate from [`JxlError::AllocLimitExceeded`] because it is a
    /// different check finding a different thing: that one prices the
    /// output frame from the header before the decoder has reserved
    /// anything, while this one is the `jxl_oxide::AllocTracker` handed to
    /// the builder refusing an *internal* buffer part-way through the decode,
    /// where the size is `jxl-oxide`'s business and is not reported out.
    /// A file can trip either without tripping the other, so collapsing
    /// them would lose which ceiling actually bit.
    ///
    /// The refusal arrives as a `jxl_grid::OutOfMemory` boxed behind one
    /// or two enum layers (`jxl_render::Error::Buffer`,
    /// `jxl_frame::Error::Buffer`); walking the source chain for it is
    /// what keeps an over-budget file off [`JxlError::Decode`], which
    /// would report it as a corrupt one. `jxl-grid` is named directly in
    /// `Cargo.toml` for exactly that downcast.
    #[error(
        "jxl: the decoder's allocation tracker refused a buffer against the \
         {max_alloc_bytes}-byte decode allocation budget; raise \
         DecodeLimits::max_alloc_bytes"
    )]
    DecoderAllocLimitExceeded {
        /// The budget in force, [`DecodeLimits::max_alloc_bytes`].
        max_alloc_bytes: u64,
    },
    /// Constructing the decoded [`Raster`] failed.
    #[error(transparent)]
    Raster(#[from] RasterError),
}

/// How the JPEG XL encoder compresses pixels (libvips `jxlsave`'s
/// `lossless` flag plus its `Q` and `distance` factors, folded into one
/// axis).
///
/// Lossless is the only representable mode because it is the only mode this
/// build can encode. `Lossy { distance }` joins the enum when there is a
/// VarDCT encoder to back it, which is a minor bump rather than a breaking
/// one thanks to `#[non_exhaustive]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Compression {
    /// Lossless modular compression (libvips `jxlsave --lossless`). The
    /// pixels round-trip exactly and there is no distance or quality
    /// factor to set.
    #[default]
    Lossless,
}

/// Options for [`Raster::encode_jxl`] (libvips `jxlsave` / `jxlsave_buffer`).
///
/// Plain, `Default`, and module-scoped, so callers write
/// `jxl::SaveOptions { compression, ..Default::default() }` and later
/// fields can be added without a breaking change.
///
/// There is no `keep` field, unlike [`crate::webp::SaveOptions`]: the
/// encoder writes a bare codestream with no box container, so there is
/// nowhere for an ICC profile, an EXIF block or an XMP packet to go.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct SaveOptions {
    /// How to compress. Defaults to [`Compression::Lossless`], the only
    /// mode with an encoder behind it.
    pub compression: Compression,
}

/// Decode JPEG XL bytes into a [`Raster`] (libvips `jxlload_buffer` at its
/// default `n = 1`).
///
/// Both container forms decode: the bare codestream, which starts `FF 0A`,
/// and the ISOBMFF form, which starts with the 12-byte `JXL ` signature
/// box. The sample carrier follows the file rather than a fixed choice,
/// the way `jxlload.c:681-696` picks one: float samples give
/// [`PixelFormat::FloatF32`], more than 8 bits per sample gives the 16-bit
/// formats, and everything else gives the 8-bit ones. The band count is
/// the colour channels plus alpha, so a greyscale file stays one band
/// where the WebP loader would give three.
///
/// `icc-profile-data`, `exif-data`, `xmp-data` and `bits-per-sample` are
/// lifted onto the raster under the field names `jxlload.c:800-825` uses,
/// so [`Raster::icc_profile`] finds a JPEG XL profile without knowing where
/// it came from. The interpretation is tagged explicitly rather than
/// inferred, because JPEG XL's band count and its *colour* channel count
/// are different questions and only the second one decides the tag.
///
/// # Animations
///
/// A multi-frame file decodes to **frame 0 only**, at one frame's size, and
/// carries `n-pages` set to the number of frames the original had, which is
/// what a default `vips jxlload` does (`jxlload.c:743-751`). Reading every
/// frame is issue #621 and needs the page model from #564; until then
/// `n-pages` is the signal that frames were left behind.
///
/// [`Raster::get_n_pages`] reads it back for anything under 10,000 frames.
/// At or above that it reports `1`, because it ports
/// `vips_image_get_n_pages`'s sanity ceiling whole (issue #635). Nothing
/// here caps the count on the way in, so a file with that many frames
/// attaches its real length and the raw value stays readable through
/// [`Raster::get_field`].
///
/// # Errors
///
/// Every JPEG XL refusal arrives as [`SourceError::Jxl`] wrapping a
/// [`JxlError`]; the two ceilings below are the shared ones every codec in
/// the crate reports the same way.
///
/// * [`JxlError::FeatureNotEnabled`] when the crate was built without the
///   `jxl` feature. Every bullet below needs the feature to be reachable
///   at all.
/// * [`JxlError::Decode`] for a malformed bitstream, and
///   [`JxlError::Truncated`] when the bytes simply run out, either before
///   the header is complete or before the first frame is.
/// * [`JxlError::CmykNotSupported`] for a file with a black ink channel.
/// * [`JxlError::AllocLimitExceeded`] when the frame the header declares
///   would exceed [`DecodeLimits::max_alloc_bytes`], and
///   [`JxlError::DecoderAllocLimitExceeded`] when `jxl-oxide`'s own
///   tracker refuses an internal buffer against the same budget.
/// * [`JxlError::UnsupportedChannelCount`] and
///   [`JxlError::ChannelCountMismatch`] for a channel count with no
///   carrier and for a rendered frame that disagrees with its header, both
///   defensive.
/// * [`JxlError::Raster`] when the decoded frame cannot be wrapped
///   (a zero-sized canvas).
/// * [`SourceError::CoordLimitExceeded`] when either declared axis exceeds
///   [`DecodeLimits::max_coord`].
/// * [`SourceError::DimensionLimitExceeded`] when `width * height` exceeds
///   [`DecodeLimits::max_pixels`].
pub fn decode_jxl(bytes: &[u8], limits: DecodeLimits) -> Result<Raster, SourceError> {
    decode(bytes, limits)
}

/// The `jxl`-feature-on body of [`decode_jxl`].
#[cfg(feature = "jxl")]
fn decode(bytes: &[u8], limits: DecodeLimits) -> Result<Raster, SourceError> {
    // The decoder's own budget, which is what stops a bomb before the
    // header geometry is even readable. `jxl-oxide` documents it as
    // advisory rather than strict, so the frame buffer is checked against
    // `max_alloc_bytes` again below.
    let tracker =
        AllocTracker::with_limit(usize::try_from(limits.max_alloc_bytes).unwrap_or(usize::MAX));
    let mut uninit = JxlImage::builder().alloc_tracker(tracker).build_uninit();

    // Two phases on purpose: feed only as far as the header, check the
    // declared geometry, and feed the frame data afterwards. Feeding the
    // whole buffer first would reserve for the frame before either ceiling
    // had been consulted, which is the bug #567 found in the WebP path.
    let consumed = uninit
        .feed_bytes(bytes)
        .map_err(|e| decode_error(e, limits))?;
    let mut image = match uninit.try_init().map_err(|e| decode_error(e, limits))? {
        InitializeResult::Initialized(image) => image,
        InitializeResult::NeedMoreData(_) => {
            return Err(JxlError::Truncated {
                expected: "the end of the image header",
            }
            .into());
        }
    };

    let (width, height) = (image.width(), image.height());
    limits.check_coord(width, height)?;
    limits.check_pixels(width, height)?;

    // Everything below is header-level and readable before a single byte
    // of frame data is fed in, which is the point of splitting the feed:
    // the carrier and the band count decide how big the frame buffer will
    // be, so the allocation budget can refuse the file before the decoder
    // reserves anything for it.
    let jxl_format = image.pixel_format();
    check_colour_space(jxl_format)?;

    // The sample carrier, chosen the way `jxlload.c:936-942` chooses one:
    // any float anywhere in the samples makes the whole image float,
    // otherwise more than 8 declared bits makes it 16-bit.
    let metadata = &image.image_header().metadata;
    let bits_per_sample = metadata.bit_depth.bits_per_sample();
    let is_float = is_float_sample(metadata.bit_depth)
        || metadata
            .ec_info
            .iter()
            .any(|ec| ec.is_alpha() && is_float_sample(ec.bit_depth));
    let bands = u32::try_from(jxl_format.channels()).unwrap_or(u32::MAX);
    let format = carrier(bands, is_float, bits_per_sample)?;

    // The allocation budget, which `check_pixels` does not imply: a
    // 1-gigapixel `max_pixels` permits an 8 GiB `Rgba16` frame against a
    // 512 MiB default budget. The `AllocTracker` above catches the same
    // thing from inside the decoder, but only once the frame data has been
    // fed in and only as an advisory; this is the ceiling with a typed
    // error behind it, and it is the same one the WebP path reports.
    //
    // The price and the comparison are the crate's, not this module's
    // (issue #632). This one used to saturate in `usize`, which makes the
    // answer depend on the target's pointer width: on a 32-bit target the
    // sample count pins at `u32::MAX` before the sample size is applied,
    // so no frame can ever be priced above `u32::MAX * sample_bytes`, or
    // about 16 GiB, however large the header says it is. A caller who has
    // raised the budget past that gets a frame accepted there that FITS
    // and OpenEXR refuse at the same geometry. `decode_alloc_bytes` widens
    // every multiplicand to `u64` first, which is the rule
    // `raster::buffer_len` states and the reason it uses `checked_mul`.
    let bytes_needed = decode_alloc_bytes(
        width,
        height,
        u64::from(bands),
        format.bytes_per_channel() as u64,
    );
    limits
        .check_alloc("JPEG XL frame buffer", bytes_needed)
        .map_err(|_| JxlError::AllocLimitExceeded {
            width,
            height,
            channels: bands,
            needed: bytes_needed,
            max_alloc_bytes: limits.max_alloc_bytes,
        })?;
    // `samples` sizes the frame buffer below and is only reached once the
    // price above has cleared the budget, so it cannot be a saturated
    // count by the time it is used.
    let samples = (width as usize)
        .saturating_mul(height as usize)
        .saturating_mul(bands as usize);

    image
        .feed_bytes(&bytes[consumed..])
        .map_err(|e| decode_error(e, limits))?;
    image.finalize().map_err(|e| decode_error(e, limits))?;
    if image.num_loaded_keyframes() == 0 {
        return Err(JxlError::Truncated {
            expected: "the first complete frame",
        }
        .into());
    }

    let frames = u32::try_from(image.num_loaded_keyframes()).unwrap_or(u32::MAX);
    let icc = image.rendered_icc();
    let exif = exif_blob(&image);
    let xmp = match image.aux_boxes().first_xml() {
        AuxBoxData::Data(x) => Some(x.to_vec()),
        _ => None,
    };

    let render = image.render_frame(0).map_err(|e| decode_error(e, limits))?;
    let mut stream = render.stream();
    // The header said how many bands there would be and the budget was
    // spent on that number, so a stream that disagrees is refused rather
    // than silently written into a buffer sized for something else.
    check_channel_count(bands, stream.channels())?;

    let data = match format.bytes_per_channel() {
        1 => {
            let mut buf = vec![0u8; samples];
            stream.write_to_buffer(&mut buf);
            buf
        }
        2 => {
            let mut buf = vec![0u16; samples];
            stream.write_to_buffer(&mut buf);
            buf.into_iter().flat_map(u16::to_ne_bytes).collect()
        }
        _ => {
            let mut buf = vec![0f32; samples];
            stream.write_to_buffer(&mut buf);
            buf.into_iter().flat_map(f32::to_ne_bytes).collect()
        }
    };

    let mut raster = Raster::new(width, height, format, data).map_err(JxlError::Raster)?;
    // Tagged rather than inferred: `Interpretation::for_format` reads the
    // band count, and a two-band greyscale-plus-alpha JXL has one colour
    // channel while a four-band float one has three. Only the colour
    // channel count decides the tag (`jxlload.c:698-737`).
    raster.meta.interpretation = Some(interpretation(jxl_format.is_grayscale(), format));
    // Upright already; see the module docs for why this is 1 rather than
    // the header's value.
    raster.meta.orientation = 1;
    raster
        .fields
        .set("icc-profile-data", MetadataValue::Blob(icc));
    if let Some(exif) = exif {
        raster.fields.set("exif-data", MetadataValue::Blob(exif));
    }
    if let Some(xmp) = xmp {
        raster.fields.set("xmp-data", MetadataValue::Blob(xmp));
    }
    raster.fields.set(
        "bits-per-sample",
        MetadataValue::Int(i64::from(bits_per_sample)),
    );
    if frames > 1 {
        raster.set_n_pages(frames);
    }
    Ok(raster)
}

/// The `jxl`-feature-off body of [`decode_jxl`]: the one [`JxlError`]
/// variant this build can produce, so a caller compiled either way sees
/// one signature and one error type and can tell "this build has no JPEG
/// XL" from "these bytes are not JPEG XL" by the variant rather than by
/// the message. [`crate::svg`] reports an `Unsupported` I/O error for the
/// same situation because it has no error enum of its own to put it on.
#[cfg(not(feature = "jxl"))]
fn decode(bytes: &[u8], limits: DecodeLimits) -> Result<Raster, SourceError> {
    let _ = (bytes, limits);
    Err(JxlError::FeatureNotEnabled.into())
}

impl Raster {
    /// Encode as lossless JPEG XL bytes (libvips `jxlsave_buffer
    /// --lossless --keep none`).
    ///
    /// Accepts the 8- and 16-bit carriers at one, two, three and four
    /// bands: [`PixelFormat::Gray8`], [`PixelFormat::Gray16`],
    /// [`PixelFormat::Rgb8`], [`PixelFormat::Rgb16`],
    /// [`PixelFormat::Rgba8`], [`PixelFormat::Rgba16`], and the two-band
    /// [`PixelFormat::Multi8`]`(2)` / [`PixelFormat::Multi16`]`(2)`, which
    /// the format calls greyscale-plus-alpha and which is what a
    /// two-band JPEG XL decodes to. Greyscale stays one band on the round
    /// trip, unlike WebP.
    ///
    /// The output is a bare codestream with no box container, so nothing
    /// attached to the raster is written: no ICC profile, no EXIF, no XMP.
    /// `vips jxlsave --keep none` writes the same form.
    ///
    /// # Errors
    ///
    /// [`EncodeError::Unsupported`] naming `"jxl"` when the crate was
    /// built without the `jxl` feature, which is the variant every
    /// format without an encoder in this build reports; otherwise
    /// [`EncodeError::Encode`] when the raster is float or has a band count
    /// the format has no spelling for (cast first; the message says so),
    /// when either axis is below [`MIN_DIMENSION`], or when the codec
    /// rejects the frame.
    pub fn encode_jxl(&self, options: SaveOptions) -> Result<Vec<u8>, EncodeError> {
        encode(self, options)
    }

    /// Save the raster to `path` as lossless JPEG XL (libvips `jxlsave`).
    ///
    /// # Errors
    ///
    /// [`SaveError::Encode`] when [`Raster::encode_jxl`] rejects the
    /// raster, or [`SaveError::Io`] when the file write fails.
    pub fn save_jxl(&self, path: &Path, options: SaveOptions) -> Result<(), SaveError> {
        let bytes = self.encode_jxl(options).map_err(encode_to_save)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// The `jxl`-feature-on body of [`Raster::encode_jxl`].
#[cfg(feature = "jxl")]
fn encode(raster: &Raster, options: SaveOptions) -> Result<Vec<u8>, EncodeError> {
    let SaveOptions { compression } = options;
    // One arm today. The `match` is deliberate: when `Lossy` lands it
    // will fail to compile here rather than silently encode losslessly.
    let Compression::Lossless = compression;

    let (colorspace, depth) = encoder_colorspace(raster.format())?;
    let (width, height) = (raster.width(), raster.height());
    if width < MIN_DIMENSION || height < MIN_DIMENSION {
        return Err(EncodeError::encode(format!(
            "zune-jpegxl cannot encode a {width}x{height} image; its floor is \
             {MIN_DIMENSION} pixels on each axis, where vips jxlsave goes down to 1x1"
        )));
    }

    // `Raster` guarantees `data().len() == width * height * bpp` by
    // construction, which is exactly what `calculate_expected_input`
    // recomputes; the debug assertion says so out loud rather than
    // leaving a `LengthMismatch` to the dependency. The floor above
    // also keeps the zero-height assertion inside `EncoderOptions`
    // unreachable.
    debug_assert_eq!(
        raster.data().len(),
        raster.stride() * height as usize,
        "a Raster's buffer is exactly its geometry"
    );
    let encoder = JxlSimpleEncoder::new(
        raster.data(),
        EncoderOptions::new(width as usize, height as usize, colorspace, depth),
    );
    let mut out = Vec::new();
    encoder
        .encode(&mut out)
        .map_err(|e| EncodeError::encode(e.to_string().trim_end().to_owned()))?;
    Ok(out)
}

/// The `jxl`-feature-off body of [`Raster::encode_jxl`]: the same
/// [`EncodeError::Unsupported`] every format without an encoder in this
/// build reports, carrying the format name so a caller matching on the
/// variant learns which one it asked for.
#[cfg(not(feature = "jxl"))]
fn encode(raster: &Raster, options: SaveOptions) -> Result<Vec<u8>, EncodeError> {
    let _ = (raster, options);
    Err(EncodeError::unsupported("jxl"))
}

/// The `.jxl` row of [`Raster::save`]'s extension route
/// (`crate::imageio::save_impl`).
///
/// Exists so the extension route carries one call rather than the error
/// mapping, and so both live next to the encoder they belong to. There is
/// no `keep_metadata` parameter because the encoder writes no metadata at
/// all: `save` and `save_stripped` produce identical bytes here, which is
/// stated in the module docs rather than hidden behind an ignored flag.
#[cfg(feature = "jxl")]
pub(crate) fn encode_jxl_for_save(raster: &Raster) -> Result<Vec<u8>, SaveError> {
    raster
        .encode_jxl(SaveOptions::default())
        .map_err(encode_to_save)
}

/// Map an encode failure onto the save error, keeping an I/O failure in its
/// own variant.
fn encode_to_save(err: EncodeError) -> SaveError {
    match err {
        EncodeError::Io(io) => SaveError::Io(io),
        other => SaveError::Encode(crate::sink::SinkError::EncodeMsg(other.to_string())),
    }
}

/// Map the codec's decode failure onto [`JxlError`].
///
/// `jxl-oxide` boxes every failure into one `dyn Error`, including the I/O
/// ones, so unlike the WebP path there is no truncation variant to keep
/// separate here; everything a parse can go wrong with lands on
/// [`JxlError::Decode`], and the message carries the whole chain because
/// every arm of `jxl-oxide`'s own `Display` writes its inner error inline.
///
/// The one failure that is not a malformed bitstream is the allocation
/// budget: the [`AllocTracker`] handed to the builder reports a refusal as
/// a `jxl_grid::OutOfMemory`, arriving boxed behind one or two enum layers
/// (`jxl_render::Error::Buffer`, `jxl_frame::Error::Buffer`). Walking the
/// source chain for it is what keeps a budget refusal on
/// [`JxlError::DecoderAllocLimitExceeded`] rather than reporting an
/// over-budget file as a corrupt one. `jxl-grid` is named directly in
/// `Cargo.toml` for exactly this downcast; if `jxl-oxide` ever moves to a
/// semver-incompatible `jxl-grid` the two copies stop unifying and the
/// downcast quietly stops matching, which is why
/// `decode_limits_are_enforced_on_the_declared_geometry` pins the mapping
/// on the variant rather than on `is_err`.
#[cfg(feature = "jxl")]
fn decode_error(
    err: Box<dyn std::error::Error + Send + Sync + 'static>,
    limits: DecodeLimits,
) -> JxlError {
    if is_out_of_memory(err.as_ref()) {
        return JxlError::DecoderAllocLimitExceeded {
            max_alloc_bytes: limits.max_alloc_bytes,
        };
    }
    JxlError::Decode {
        message: err.to_string(),
    }
}

/// Whether an error, or anything in its source chain, is the allocation
/// tracker refusing a request.
#[cfg(feature = "jxl")]
fn is_out_of_memory(err: &(dyn std::error::Error + 'static)) -> bool {
    let mut cause = Some(err);
    while let Some(e) = cause {
        if e.downcast_ref::<jxl_grid::OutOfMemory>().is_some() {
            return true;
        }
        cause = e.source();
    }
    false
}

/// Refuse a colour space this loader has no carrier and no tag for, which
/// today means any file with a black ink channel.
///
/// A seam rather than an inline `if`, because nothing in this build can
/// write a CMYK JPEG XL to test it against (`vips jxlsave` converts a
/// `cmyk` image to sRGB on the way in), so the refusal is reachable from a
/// test only through the function.
#[cfg(feature = "jxl")]
fn check_colour_space(format: jxl_oxide::PixelFormat) -> Result<(), JxlError> {
    if format.has_black() {
        return Err(JxlError::CmykNotSupported {
            pixel_format: format!("{format:?}"),
        });
    }
    Ok(())
}

/// Refuse a rendered frame whose channel count is not the one the header
/// declared and the allocation budget was spent on.
///
/// A seam for the same reason as [`check_colour_space`]: `jxl-oxide`
/// renders what the header said, so the disagreement is not reachable from
/// a file, and a check nothing can exercise is a check nobody notices
/// breaking.
#[cfg(feature = "jxl")]
fn check_channel_count(declared: u32, rendered: u32) -> Result<(), JxlError> {
    if declared != rendered {
        return Err(JxlError::ChannelCountMismatch { declared, rendered });
    }
    Ok(())
}

/// Whether a declared bit depth stores floating-point samples
/// (`jxlload.c:936-937` asks libjxl the same question as
/// `exponent_bits_per_sample > 0`).
#[cfg(feature = "jxl")]
fn is_float_sample(depth: jxl_oxide::image::BitDepth) -> bool {
    matches!(depth, jxl_oxide::image::BitDepth::FloatSample { .. })
}

/// The pixel format for a band count and sample carrier, or the reason
/// there is none.
///
/// Band counts 1, 3 and 4 land on the named variants and everything else on
/// the multiband carriers, which is what `PixelFormat::canonical` does; the
/// float arm is spelled separately because a four-band float canonicalises
/// to [`PixelFormat::RgbaF32`] rather than to `FloatF32(4)`.
#[cfg(feature = "jxl")]
fn carrier(bands: u32, is_float: bool, bits_per_sample: u32) -> Result<PixelFormat, JxlError> {
    let n = u16::try_from(bands).ok().and_then(NonZeroU16::new).ok_or(
        JxlError::UnsupportedChannelCount {
            channels: bands,
            max: u16::MAX as u32,
        },
    )?;
    Ok(match (is_float, bits_per_sample, n.get()) {
        (true, _, 4) => PixelFormat::RgbaF32,
        (true, _, _) => PixelFormat::FloatF32(n),
        (false, 9.., 1) => PixelFormat::Gray16,
        (false, 9.., 3) => PixelFormat::Rgb16,
        (false, 9.., 4) => PixelFormat::Rgba16,
        (false, 9.., _) => PixelFormat::Multi16(n),
        (false, _, 1) => PixelFormat::Gray8,
        (false, _, 3) => PixelFormat::Rgb8,
        (false, _, 4) => PixelFormat::Rgba8,
        (false, _, _) => PixelFormat::Multi8(n),
    })
}

/// The interpretation tag `jxlload.c:698-737` assigns, which reads the
/// *colour* channel count and the sample carrier and ignores the band
/// count entirely.
#[cfg(feature = "jxl")]
fn interpretation(is_grayscale: bool, format: PixelFormat) -> Interpretation {
    let bytes = format.bytes_per_channel();
    match (is_grayscale, bytes) {
        // One colour channel: `b-w` for uchar and float alike, `grey16`
        // for ushort.
        (true, 2) => Interpretation::Grey16,
        (true, _) => Interpretation::Bw,
        // Three colour channels: sRGB, rgb16, or linear-light scRGB for
        // the float carrier.
        (false, 4) => Interpretation::ScRgb,
        (false, 2) => Interpretation::Rgb16,
        (false, _) => Interpretation::Srgb,
    }
}

/// The `exif-data` blob for an image, or `None` when there is no readable
/// `Exif` box.
///
/// The transform is `jxlload.c:650-658`: read the big-endian 4-byte
/// tiff_header_offset, skip that many bytes of the payload, and put
/// `Exif\0\0` back on the front. A box `jxl-oxide` refuses to parse costs
/// the blob and not the image; see the module docs for the two shapes where
/// that diverges from vips.
#[cfg(feature = "jxl")]
fn exif_blob(image: &JxlImage) -> Option<Vec<u8>> {
    let AuxBoxData::Data(exif) = image.aux_boxes().first_exif().ok()? else {
        return None;
    };
    let payload = exif.payload();
    let offset = usize::try_from(exif.tiff_header_offset()).ok()?;
    let tiff = payload.get(offset..)?;
    let mut blob = Vec::with_capacity(EXIF_PREFIX.len() + tiff.len());
    blob.extend_from_slice(EXIF_PREFIX);
    blob.extend_from_slice(tiff);
    Some(blob)
}

/// The encoder colour space and depth for a raster, or the reason there is
/// none.
///
/// `zune-jpegxl` holds one to four channels at 8 or 16 bits
/// (`errors.rs:SUPPORTED_COLORSPACES` and `SUPPORTED_DEPTHS`), which is
/// every integer carrier libviprs has below five bands. Float is refused
/// rather than quantised, matching [`crate::sink::encode_png`] and
/// [`Raster::encode_radiance`].
#[cfg(feature = "jxl")]
fn encoder_colorspace(format: PixelFormat) -> Result<(ColorSpace, BitDepth), EncodeError> {
    let two = NonZeroU16::new(2).expect("2 is non-zero");
    match format {
        PixelFormat::Gray8 => Ok((ColorSpace::Luma, BitDepth::Eight)),
        PixelFormat::Gray16 => Ok((ColorSpace::Luma, BitDepth::Sixteen)),
        PixelFormat::Rgb8 => Ok((ColorSpace::RGB, BitDepth::Eight)),
        PixelFormat::Rgb16 => Ok((ColorSpace::RGB, BitDepth::Sixteen)),
        PixelFormat::Rgba8 => Ok((ColorSpace::RGBA, BitDepth::Eight)),
        PixelFormat::Rgba16 => Ok((ColorSpace::RGBA, BitDepth::Sixteen)),
        PixelFormat::Multi8(n) if n == two => Ok((ColorSpace::LumaA, BitDepth::Eight)),
        PixelFormat::Multi16(n) if n == two => Ok((ColorSpace::LumaA, BitDepth::Sixteen)),
        PixelFormat::RgbaF32 | PixelFormat::FloatF32(_) => Err(EncodeError::encode(format!(
            "zune-jpegxl encodes 8- and 16-bit integer samples and {format:?} is float; \
             cast to an integer format first, so the quantisation is yours rather than \
             the encoder's (vips jxlsave writes float samples natively)"
        ))),
        other => Err(EncodeError::encode(format!(
            "zune-jpegxl holds 1, 2, 3 or 4 bands of 8- or 16-bit samples and {other:?} \
             has no such spelling; cast to Gray8, Rgb8 or Rgba8 first"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Named explicitly rather than taken from the glob: the parent's
    // import is behind the feature and `ramp_rgb` is not.
    use crate::pixel::PixelFormat;
    #[cfg(feature = "jxl")]
    use crate::source::decode_bytes_with_limits;

    // -----------------------------------------------------------------
    // Oracle fixtures. Every byte below came out of vips 8.18.4 or out of
    // a container this lane hand-built and then asked vips to read; the
    // commands are in `oracle-captures/foreign-jxl/commands.sh` and the
    // expected pixels in its `oracle.json`.
    // -----------------------------------------------------------------

    /// `vips jxlsave --lossless --keep none` on the 4x3 sRGB raster in
    /// [`ramp_rgb`], captured verbatim. A bare codestream, 71 bytes.
    const LOSSLESS_RGB: [u8; 71] = [
        0xff, 0x0a, 0x10, 0x30, 0x10, 0x09, 0x08, 0x00, 0x01, 0x00, 0xec, 0x00, 0x4b, 0x18, 0x8b,
        0x15, 0xc2, 0x09, 0x32, 0x37, 0x38, 0xa6, 0xeb, 0x02, 0xb1, 0x00, 0x94, 0xce, 0x40, 0xdf,
        0x67, 0x8c, 0x87, 0xae, 0x85, 0x66, 0x8e, 0x97, 0xf4, 0x88, 0xc3, 0xf0, 0x2c, 0x95, 0xc2,
        0x99, 0x42, 0x78, 0xd8, 0xad, 0x48, 0xdc, 0x3f, 0x43, 0x19, 0xe2, 0x2f, 0x34, 0x83, 0xde,
        0xca, 0x62, 0x52, 0x93, 0x08, 0x3c, 0x38, 0x51, 0x1e, 0x9d, 0x04,
    ];

    /// The same 4x3 raster with an alpha ramp, `vips jxlsave --lossless
    /// --keep none`. 91 bytes.
    #[cfg(feature = "jxl")]
    const LOSSLESS_RGBA: [u8; 91] = [
        0xff, 0x0a, 0x10, 0x30, 0xb0, 0x12, 0x08, 0x00, 0x10, 0x00, 0x3c, 0x01, 0x4b, 0x18, 0x8b,
        0x15, 0x52, 0x5c, 0xd6, 0x6f, 0xf7, 0x26, 0x91, 0x2f, 0xd4, 0xda, 0xd2, 0xba, 0x9f, 0xcd,
        0xab, 0x9d, 0x31, 0x17, 0x6a, 0x17, 0x90, 0x05, 0x80, 0xda, 0x21, 0x73, 0x1c, 0x2a, 0xc1,
        0xf5, 0x2d, 0x40, 0x25, 0x3b, 0x7c, 0x11, 0xcb, 0x6d, 0x9e, 0x34, 0xeb, 0x75, 0x91, 0x81,
        0xe7, 0xee, 0x20, 0xe1, 0x4f, 0x6b, 0xa2, 0x36, 0x5e, 0x9a, 0x6f, 0x84, 0x4c, 0x50, 0x2a,
        0x67, 0x3e, 0x5c, 0xe6, 0xac, 0x01, 0xf5, 0x56, 0xa1, 0xa4, 0x29, 0xe2, 0x95, 0x09, 0xde,
        0x69,
    ];

    /// The luminance band of the same ramp as a `b-w` image, `vips jxlsave
    /// --lossless --keep none`. 34 bytes, and it stays one band.
    #[cfg(feature = "jxl")]
    const LOSSLESS_GREY: [u8; 34] = [
        0xff, 0x0a, 0x10, 0x30, 0x10, 0x14, 0x37, 0x02, 0x08, 0x00, 0x01, 0x00, 0x50, 0x00, 0x4b,
        0x18, 0x8b, 0x15, 0x42, 0x19, 0x36, 0x6e, 0x53, 0xcd, 0xd2, 0xd3, 0xd3, 0xc9, 0x67, 0x51,
        0x50, 0x26, 0xd0, 0x02,
    ];

    /// A 4x3 `rgb16` ramp, `vips jxlsave --lossless --keep none`. 138
    /// bytes, and the samples come back at their full 16-bit range.
    #[cfg(feature = "jxl")]
    const LOSSLESS_RGB16: [u8; 138] = [
        0x00, 0x00, 0x00, 0x0c, 0x4a, 0x58, 0x4c, 0x20, 0x0d, 0x0a, 0x87, 0x0a, 0x00, 0x00, 0x00,
        0x14, 0x66, 0x74, 0x79, 0x70, 0x6a, 0x78, 0x6c, 0x20, 0x00, 0x00, 0x00, 0x00, 0x6a, 0x78,
        0x6c, 0x20, 0x00, 0x00, 0x00, 0x09, 0x6a, 0x78, 0x6c, 0x6c, 0x0a, 0x00, 0x00, 0x00, 0x61,
        0x6a, 0x78, 0x6c, 0x63, 0xff, 0x0a, 0x10, 0x30, 0xfc, 0x40, 0x02, 0x08, 0x00, 0x01, 0x00,
        0x30, 0x01, 0x4b, 0x18, 0x8b, 0x15, 0x01, 0x12, 0x9e, 0x18, 0x6a, 0x9b, 0xe9, 0x13, 0x08,
        0xf9, 0x27, 0x4e, 0xc3, 0xda, 0x05, 0x22, 0x01, 0x78, 0xea, 0x7f, 0x61, 0xdf, 0xeb, 0xe9,
        0xb7, 0xad, 0xec, 0x19, 0xf6, 0x6a, 0x74, 0x0a, 0xba, 0x07, 0xfb, 0x83, 0xc2, 0x19, 0x2d,
        0xda, 0x11, 0x3c, 0xf8, 0x5c, 0x93, 0x02, 0xdc, 0x00, 0xb4, 0x40, 0x19, 0x61, 0x80, 0x8c,
        0x24, 0x9b, 0xad, 0xb3, 0xb2, 0x56, 0x78, 0xf7, 0x92, 0x54, 0xe4, 0x72, 0xaa, 0x80, 0x94,
        0x49, 0x88, 0x13,
    ];

    /// A 4x3 `scrgb` float ramp, `vips jxlsave --lossless --keep none`.
    /// 192 bytes, and every dyadic value survives exactly.
    #[cfg(feature = "jxl")]
    const LOSSLESS_F32: [u8; 192] = [
        0x00, 0x00, 0x00, 0x0c, 0x4a, 0x58, 0x4c, 0x20, 0x0d, 0x0a, 0x87, 0x0a, 0x00, 0x00, 0x00,
        0x14, 0x66, 0x74, 0x79, 0x70, 0x6a, 0x78, 0x6c, 0x20, 0x00, 0x00, 0x00, 0x00, 0x6a, 0x78,
        0x6c, 0x20, 0x00, 0x00, 0x00, 0x09, 0x6a, 0x78, 0x6c, 0x6c, 0x0a, 0x00, 0x00, 0x00, 0x97,
        0x6a, 0x78, 0x6c, 0x63, 0xff, 0x0a, 0x10, 0x30, 0x72, 0x00, 0x45, 0x8b, 0x08, 0x00, 0x01,
        0x00, 0x04, 0x02, 0x4b, 0x38, 0x69, 0x98, 0xca, 0x83, 0xf7, 0x2b, 0x28, 0x48, 0xa1, 0xc5,
        0xc1, 0x39, 0xf6, 0x96, 0x87, 0x0c, 0x79, 0x76, 0x6e, 0x00, 0x00, 0x80, 0xbe, 0x00, 0x00,
        0x00, 0x01, 0x00, 0x80, 0x01, 0x00, 0x00, 0xf6, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x28, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x06, 0x00,
        0x00, 0xd0, 0x1f, 0x00, 0x00, 0x52, 0x00, 0x00, 0x5c, 0x00, 0x00, 0xa0, 0x00, 0x00, 0xc0,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x80, 0x01, 0x00, 0x00, 0xf6, 0x07, 0x00, 0x00, 0x11, 0x00, 0x00,
        0x16, 0x00, 0x00, 0x28, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x04, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];

    /// `vips jxlsave` at the default distance on the same 4x3 ramp: the
    /// VarDCT float path, which libviprs decodes but cannot write.
    #[cfg(feature = "jxl")]
    const LOSSY_RGB: [u8; 103] = [
        0xff, 0x0a, 0x10, 0xb0, 0x01, 0x00, 0x1c, 0x48, 0x00, 0x70, 0x01, 0xf3, 0x43, 0x13, 0x00,
        0x80, 0x0a, 0x95, 0x51, 0xc6, 0x0d, 0x5e, 0xce, 0xf5, 0x7c, 0xf9, 0xa1, 0xc3, 0x62, 0xda,
        0xc6, 0x75, 0x86, 0xb6, 0xda, 0xb6, 0xb0, 0x84, 0xb1, 0xd9, 0x5e, 0x18, 0x24, 0x24, 0x1c,
        0x35, 0xcc, 0xa9, 0x66, 0x4e, 0x21, 0x51, 0x43, 0xab, 0x28, 0x15, 0x52, 0xe0, 0x0e, 0xd7,
        0x2a, 0xcd, 0x18, 0xdb, 0x46, 0x4a, 0xe1, 0x1f, 0xf8, 0x87, 0x89, 0x1e, 0xa0, 0xba, 0xff,
        0xc1, 0x90, 0xa4, 0xf7, 0xfd, 0xac, 0x12, 0x69, 0x02, 0x8a, 0xfa, 0x46, 0x02, 0x26, 0x7c,
        0xb6, 0x9f, 0x7d, 0xc7, 0xf5, 0xad, 0x91, 0xf5, 0xf7, 0x06, 0x1e, 0x60, 0x69,
    ];

    /// `vips jxlsave --lossless --page-height 3` on a 4x9 toilet roll: three
    /// 4x3 frames whose frame 0 is the [`LOSSLESS_RGB`] image. vips reports
    /// `n-pages: 3` and loads 4x3 by default.
    #[cfg(feature = "jxl")]
    const ANIM3: [u8; 237] = [
        0xff, 0x0a, 0x10, 0x30, 0xc1, 0x00, 0x62, 0x02, 0x08, 0x00, 0xff, 0xff, 0xff, 0xff, 0x03,
        0x00, 0xec, 0x00, 0x4b, 0x18, 0x8b, 0x15, 0xc2, 0x09, 0x32, 0x37, 0x38, 0xa6, 0xeb, 0x02,
        0xb1, 0x00, 0x94, 0xce, 0x40, 0xdf, 0x67, 0x8c, 0x87, 0xae, 0x85, 0x66, 0x8e, 0x97, 0xf4,
        0x88, 0xc3, 0xf0, 0x2c, 0x95, 0xc2, 0x99, 0x42, 0x78, 0xd8, 0xad, 0x48, 0xdc, 0x3f, 0x43,
        0x19, 0xe2, 0x2f, 0x34, 0x83, 0xde, 0xca, 0x62, 0x52, 0x93, 0x08, 0x3c, 0x38, 0x51, 0x1e,
        0x9d, 0x04, 0x08, 0x00, 0xff, 0xff, 0xff, 0xff, 0x03, 0x00, 0x14, 0x01, 0x4b, 0x18, 0x8b,
        0x15, 0x52, 0x5c, 0xf5, 0x76, 0x89, 0x12, 0x77, 0x4a, 0x3c, 0x25, 0x5b, 0x76, 0xc9, 0x5a,
        0xb6, 0x6e, 0xb7, 0x76, 0x81, 0x60, 0x00, 0x96, 0xd0, 0x39, 0xd0, 0x39, 0xfc, 0x88, 0x1f,
        0x5d, 0xc0, 0x47, 0x85, 0x1d, 0x99, 0x86, 0x3b, 0x7a, 0xfa, 0x44, 0x8d, 0x85, 0xdd, 0x47,
        0xa7, 0xd3, 0x45, 0xa5, 0xeb, 0x6a, 0xa1, 0xb5, 0x3d, 0xa2, 0xdf, 0x66, 0xce, 0x40, 0x88,
        0xe8, 0x30, 0x53, 0xde, 0x46, 0x00, 0x08, 0x00, 0xff, 0xff, 0xff, 0xff, 0x07, 0x00, 0x1c,
        0x01, 0x4b, 0x18, 0x8b, 0x15, 0x52, 0x5c, 0x76, 0x37, 0xf5, 0x11, 0x91, 0x80, 0x1a, 0x7d,
        0x16, 0x68, 0x87, 0x47, 0xd3, 0x1e, 0x69, 0x7a, 0xa4, 0x17, 0x08, 0x06, 0xa0, 0xf6, 0x10,
        0xd5, 0x19, 0x0d, 0x8c, 0x54, 0xeb, 0x22, 0x34, 0xc8, 0xa1, 0xca, 0x5f, 0xba, 0x9f, 0x80,
        0xff, 0x48, 0xd9, 0x1b, 0x70, 0x57, 0x2e, 0x34, 0xc4, 0xee, 0xfe, 0xc8, 0xb6, 0x1c, 0xb4,
        0x32, 0x73, 0x2a, 0x7f, 0x03, 0xe0, 0x4d, 0xb4, 0x21, 0x41, 0x11, 0x10,
    ];

    /// `vips jxlsave --lossless --page-height 3 --strip` on a 4x15
    /// toilet roll of five flat greys: FIVE 4x3 frames as a bare
    /// codestream. vips 8.18.6 reports `n-pages: 5` and loads 4x3.
    ///
    /// Five is the point, for the reason `ANIM3` cannot carry: three is
    /// also that fixture's band count and one of its axes, so a wrong
    /// number under `n-pages` would read as right (issue #635).
    #[cfg(feature = "jxl")]
    const ANIM5: [u8; 272] = [
        0xff, 0x0a, 0x10, 0x30, 0xc1, 0x00, 0x72, 0x02, 0x00, 0x13, 0xf8, 0xff, 0xff, 0xff, 0x1f,
        0x04, 0xa8, 0x00, 0xf3, 0x43, 0x13, 0x00, 0x80, 0x0a, 0x95, 0x51, 0xc6, 0x0d, 0x5e, 0xce,
        0xf5, 0x7c, 0xf9, 0xa1, 0xc3, 0x62, 0xda, 0xc6, 0x75, 0x86, 0xb6, 0xda, 0xb6, 0xb0, 0x84,
        0xb1, 0xd9, 0x5e, 0x18, 0x24, 0x24, 0x94, 0x13, 0x7d, 0x33, 0xc6, 0x20, 0x50, 0x49, 0x02,
        0x00, 0x13, 0xf8, 0xff, 0xff, 0xff, 0x1f, 0x04, 0xac, 0x00, 0xf3, 0x43, 0x13, 0x00, 0x80,
        0x0a, 0x95, 0x51, 0xc6, 0x0d, 0x5e, 0xce, 0xf5, 0x7c, 0xf9, 0xa1, 0xc3, 0x62, 0xda, 0xc6,
        0x75, 0x86, 0xb6, 0xda, 0xb6, 0xb0, 0x84, 0xb1, 0xd9, 0x5e, 0x18, 0x24, 0x24, 0x9c, 0x24,
        0x54, 0x36, 0x43, 0x31, 0x48, 0x54, 0x92, 0x00, 0x00, 0x13, 0xf8, 0xff, 0xff, 0xff, 0x1f,
        0x04, 0xac, 0x00, 0xf3, 0x43, 0x13, 0x00, 0x80, 0x0a, 0x95, 0x51, 0xc6, 0x0d, 0x5e, 0xce,
        0xf5, 0x7c, 0xf9, 0xa1, 0xc3, 0x62, 0xda, 0xc6, 0x75, 0x86, 0xb6, 0xda, 0xb6, 0xb0, 0x84,
        0xb1, 0xd9, 0x5e, 0x18, 0x24, 0x24, 0x1c, 0x25, 0x94, 0x36, 0xf3, 0x60, 0x90, 0xa8, 0x24,
        0x01, 0x00, 0x13, 0xf8, 0xff, 0xff, 0xff, 0x1f, 0x04, 0xac, 0x00, 0xf3, 0x43, 0x13, 0x00,
        0x80, 0x0a, 0x95, 0x51, 0xc6, 0x0d, 0x5e, 0xce, 0xf5, 0x7c, 0xf9, 0xa1, 0xc3, 0x62, 0xda,
        0xc6, 0x75, 0x86, 0xb6, 0xda, 0xb6, 0xb0, 0x84, 0xb1, 0xd9, 0x5e, 0x18, 0x24, 0x24, 0x1c,
        0x25, 0x94, 0x36, 0x73, 0x64, 0x90, 0xa8, 0x24, 0x01, 0x00, 0x13, 0xf8, 0xff, 0xff, 0xff,
        0x3f, 0x01, 0xac, 0x00, 0xf3, 0x43, 0x13, 0x00, 0x80, 0x0a, 0x95, 0x51, 0xc6, 0x0d, 0x5e,
        0xce, 0xf5, 0x7c, 0xf9, 0xa1, 0xc3, 0x62, 0xda, 0xc6, 0x75, 0x86, 0xb6, 0xda, 0xb6, 0xb0,
        0x84, 0xb1, 0xd9, 0x5e, 0x18, 0x24, 0x24, 0x1c, 0x25, 0x94, 0x36, 0xc3, 0x67, 0x10, 0xa8,
        0x24, 0x01,
    ];

    /// A hand-built container: the [`LOSSLESS_RGB`] codestream plus an
    /// `Exif` box at tiff_header_offset 0 and an `xml ` box. vips reports
    /// a 16-byte `exif-data` and a 37-byte `xmp-data` for it.
    #[cfg(feature = "jxl")]
    const META_OFF0: [u8; 178] = [
        0x00, 0x00, 0x00, 0x0c, 0x4a, 0x58, 0x4c, 0x20, 0x0d, 0x0a, 0x87, 0x0a, 0x00, 0x00, 0x00,
        0x14, 0x66, 0x74, 0x79, 0x70, 0x6a, 0x78, 0x6c, 0x20, 0x00, 0x00, 0x00, 0x00, 0x6a, 0x78,
        0x6c, 0x20, 0x00, 0x00, 0x00, 0x16, 0x45, 0x78, 0x69, 0x66, 0x00, 0x00, 0x00, 0x00, 0x49,
        0x49, 0x2a, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2d, 0x78, 0x6d,
        0x6c, 0x20, 0x3c, 0x78, 0x3a, 0x78, 0x6d, 0x70, 0x6d, 0x65, 0x74, 0x61, 0x20, 0x78, 0x6d,
        0x6c, 0x6e, 0x73, 0x3a, 0x78, 0x3d, 0x22, 0x61, 0x64, 0x6f, 0x62, 0x65, 0x3a, 0x6e, 0x73,
        0x3a, 0x6d, 0x65, 0x74, 0x61, 0x2f, 0x22, 0x2f, 0x3e, 0x00, 0x00, 0x00, 0x4f, 0x6a, 0x78,
        0x6c, 0x63, 0xff, 0x0a, 0x10, 0x30, 0x10, 0x09, 0x08, 0x00, 0x01, 0x00, 0xec, 0x00, 0x4b,
        0x18, 0x8b, 0x15, 0xc2, 0x09, 0x32, 0x37, 0x38, 0xa6, 0xeb, 0x02, 0xb1, 0x00, 0x94, 0xce,
        0x40, 0xdf, 0x67, 0x8c, 0x87, 0xae, 0x85, 0x66, 0x8e, 0x97, 0xf4, 0x88, 0xc3, 0xf0, 0x2c,
        0x95, 0xc2, 0x99, 0x42, 0x78, 0xd8, 0xad, 0x48, 0xdc, 0x3f, 0x43, 0x19, 0xe2, 0x2f, 0x34,
        0x83, 0xde, 0xca, 0x62, 0x52, 0x93, 0x08, 0x3c, 0x38, 0x51, 0x1e, 0x9d, 0x04,
    ];

    /// The same TIFF block behind a tiff_header_offset of 6, with six bytes
    /// of padding in front of it. vips reports the identical 16-byte blob.
    #[cfg(feature = "jxl")]
    const META_OFF6: [u8; 139] = [
        0x00, 0x00, 0x00, 0x0c, 0x4a, 0x58, 0x4c, 0x20, 0x0d, 0x0a, 0x87, 0x0a, 0x00, 0x00, 0x00,
        0x14, 0x66, 0x74, 0x79, 0x70, 0x6a, 0x78, 0x6c, 0x20, 0x00, 0x00, 0x00, 0x00, 0x6a, 0x78,
        0x6c, 0x20, 0x00, 0x00, 0x00, 0x1c, 0x45, 0x78, 0x69, 0x66, 0x00, 0x00, 0x00, 0x06, 0x50,
        0x41, 0x44, 0x50, 0x41, 0x44, 0x49, 0x49, 0x2a, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x4f, 0x6a, 0x78, 0x6c, 0x63, 0xff, 0x0a, 0x10, 0x30, 0x10, 0x09, 0x08,
        0x00, 0x01, 0x00, 0xec, 0x00, 0x4b, 0x18, 0x8b, 0x15, 0xc2, 0x09, 0x32, 0x37, 0x38, 0xa6,
        0xeb, 0x02, 0xb1, 0x00, 0x94, 0xce, 0x40, 0xdf, 0x67, 0x8c, 0x87, 0xae, 0x85, 0x66, 0x8e,
        0x97, 0xf4, 0x88, 0xc3, 0xf0, 0x2c, 0x95, 0xc2, 0x99, 0x42, 0x78, 0xd8, 0xad, 0x48, 0xdc,
        0x3f, 0x43, 0x19, 0xe2, 0x2f, 0x34, 0x83, 0xde, 0xca, 0x62, 0x52, 0x93, 0x08, 0x3c, 0x38,
        0x51, 0x1e, 0x9d, 0x04,
    ];

    /// An `Exif` box whose tiff_header_offset (999) runs past its 10-byte
    /// payload. vips fails the whole load on this file; libviprs drops the
    /// blob and keeps the pixels.
    #[cfg(feature = "jxl")]
    const META_BAD_OFFSET: [u8; 133] = [
        0x00, 0x00, 0x00, 0x0c, 0x4a, 0x58, 0x4c, 0x20, 0x0d, 0x0a, 0x87, 0x0a, 0x00, 0x00, 0x00,
        0x14, 0x66, 0x74, 0x79, 0x70, 0x6a, 0x78, 0x6c, 0x20, 0x00, 0x00, 0x00, 0x00, 0x6a, 0x78,
        0x6c, 0x20, 0x00, 0x00, 0x00, 0x16, 0x45, 0x78, 0x69, 0x66, 0x00, 0x00, 0x03, 0xe7, 0x49,
        0x49, 0x2a, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x4f, 0x6a, 0x78,
        0x6c, 0x63, 0xff, 0x0a, 0x10, 0x30, 0x10, 0x09, 0x08, 0x00, 0x01, 0x00, 0xec, 0x00, 0x4b,
        0x18, 0x8b, 0x15, 0xc2, 0x09, 0x32, 0x37, 0x38, 0xa6, 0xeb, 0x02, 0xb1, 0x00, 0x94, 0xce,
        0x40, 0xdf, 0x67, 0x8c, 0x87, 0xae, 0x85, 0x66, 0x8e, 0x97, 0xf4, 0x88, 0xc3, 0xf0, 0x2c,
        0x95, 0xc2, 0x99, 0x42, 0x78, 0xd8, 0xad, 0x48, 0xdc, 0x3f, 0x43, 0x19, 0xe2, 0x2f, 0x34,
        0x83, 0xde, 0xca, 0x62, 0x52, 0x93, 0x08, 0x3c, 0x38, 0x51, 0x1e, 0x9d, 0x04,
    ];

    /// The twelve RGB triples `vips getpoint` prints for every fixture that
    /// carries the lossless 8-bit ramp. Identical to the WebP capture's,
    /// because both areas are built from the same generator.
    #[cfg(feature = "jxl")]
    const RAMP_PIXELS: [[u8; 3]; 12] = [
        [0, 0, 0],
        [61, 97, 29],
        [122, 194, 58],
        [183, 35, 87],
        [13, 151, 211],
        [74, 248, 240],
        [135, 89, 13],
        [196, 186, 42],
        [26, 46, 166],
        [87, 143, 195],
        [148, 240, 224],
        [209, 81, 253],
    ];

    /// The alpha band `vips getpoint` prints for [`LOSSLESS_RGBA`].
    #[cfg(feature = "jxl")]
    const RAMP_ALPHA: [u8; 12] = [0, 85, 170, 255, 40, 125, 210, 39, 80, 165, 250, 79];

    /// The twelve 16-bit triples `vips getpoint` prints for
    /// [`LOSSLESS_RGB16`].
    #[cfg(feature = "jxl")]
    const RAMP16_PIXELS: [[u16; 3]; 12] = [
        [0, 0, 0],
        [1013, 4099, 7919],
        [2026, 8198, 15838],
        [3039, 12297, 23757],
        [3039, 12297, 23757],
        [4052, 16396, 31676],
        [5065, 20495, 39595],
        [6078, 24594, 47514],
        [6078, 24594, 47514],
        [7091, 28693, 55433],
        [8104, 32792, 63352],
        [9117, 36891, 5735],
    ];

    /// The `Exif\0\0` blob vips reports for [`META_OFF0`] and
    /// [`META_OFF6`] alike: the prefix plus a ten-byte little-endian TIFF
    /// header with an empty IFD.
    #[cfg(feature = "jxl")]
    const EXIF_BLOB: &[u8] = b"Exif\x00\x00II*\x00\x08\x00\x00\x00\x00\x00";

    /// The XMP packet vips reports for [`META_OFF0`], verbatim from the
    /// `xml ` box.
    #[cfg(feature = "jxl")]
    const XMP_PACKET: &[u8] = b"<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"/>";

    /// The 4x3 sRGB ramp every 8-bit fixture above was written from.
    fn ramp_rgb() -> Raster {
        let mut data = Vec::with_capacity(4 * 3 * 3);
        for y in 0..3u32 {
            for x in 0..4u32 {
                data.push(((x * 61 + y * 13) % 256) as u8);
                data.push(((x * 97 + y * 151) % 256) as u8);
                data.push(((x * 29 + y * 211) % 256) as u8);
            }
        }
        Raster::new(4, 3, PixelFormat::Rgb8, data).unwrap()
    }

    /// The same ramp with a fourth, independent alpha channel.
    #[cfg(feature = "jxl")]
    fn ramp_rgba() -> Raster {
        let mut data = Vec::with_capacity(4 * 3 * 4);
        for y in 0..3u32 {
            for x in 0..4u32 {
                data.push(((x * 61 + y * 13) % 256) as u8);
                data.push(((x * 97 + y * 151) % 256) as u8);
                data.push(((x * 29 + y * 211) % 256) as u8);
                data.push(((x * 85 + y * 40) % 256) as u8);
            }
        }
        Raster::new(4, 3, PixelFormat::Rgba8, data).unwrap()
    }

    /// The 16-bit ramp the `rgb16` fixture was written from, as
    /// native-endian `u16` samples.
    #[cfg(feature = "jxl")]
    fn ramp_rgb16() -> Raster {
        let mut data = Vec::with_capacity(4 * 3 * 3 * 2);
        for y in 0..3u32 {
            for x in 0..4u32 {
                for m in [1013u32, 4099, 7919] {
                    let v = ((x * m + y * (m * 3)) % 65536) as u16;
                    data.extend_from_slice(&v.to_ne_bytes());
                }
            }
        }
        Raster::new(4, 3, PixelFormat::Rgb16, data).unwrap()
    }

    /// Read every pixel of `raster` in raster order as bytes.
    #[cfg(feature = "jxl")]
    fn pixels(raster: &Raster) -> Vec<Vec<u8>> {
        (0..raster.height())
            .flat_map(|y| (0..raster.width()).map(move |x| (x, y)))
            .map(|(x, y)| {
                raster
                    .getpoint(x, y)
                    .iter()
                    .map(|s| *s as u8)
                    .collect::<Vec<u8>>()
            })
            .collect()
    }

    /// Read every sample of `raster` in raster order as `f64`.
    #[cfg(feature = "jxl")]
    fn samples(raster: &Raster) -> Vec<Vec<f64>> {
        (0..raster.height())
            .flat_map(|y| (0..raster.width()).map(move |x| (x, y)))
            .map(|(x, y)| raster.getpoint(x, y))
            .collect()
    }

    /// The blob attached under `name`, or a panic naming what was there
    /// instead.
    #[cfg(feature = "jxl")]
    fn blob(raster: &Raster, name: &str) -> Vec<u8> {
        match raster.get_field(name) {
            Some(MetadataValue::Blob(b)) => b.clone(),
            other => panic!("{name} should be a blob, got {other:?}"),
        }
    }

    /**
     * Tests that a lossless JPEG XL written by vips decodes to exactly the
     * pixels vips reads back out of it, so the modular path is pinned to
     * the reference decoder rather than to itself, and that the four header
     * fields `jxlload` attaches come with it. Works by decoding the 71-byte
     * `--lossless --keep none` capture and comparing every pixel to the
     * `vips getpoint` output recorded beside it.
     * Input: `LOSSLESS_RGB` -> Output: 4x3 `Rgb8`, pixels equal to
     * `RAMP_PIXELS`, `bits-per-sample` 8, an `icc-profile-data` blob,
     * interpretation `srgb`, and no `n-pages` field, which is what
     * `vipsheader -a` reports for the same file.
     */
    #[cfg(feature = "jxl")]
    #[test]
    fn lossless_decode_matches_vips_getpoint() {
        let raster = decode_bytes_with_limits(&LOSSLESS_RGB, DecodeLimits::default())
            .expect("the vips lossless capture decodes");
        assert_eq!((raster.width(), raster.height()), (4, 3));
        assert_eq!(raster.format(), PixelFormat::Rgb8);
        assert_eq!(pixels(&raster), RAMP_PIXELS.map(Vec::from).to_vec());
        assert_eq!(raster.interpretation(), Interpretation::Srgb);
        assert_eq!(
            raster.get_field("bits-per-sample"),
            Some(MetadataValue::Int(8))
        );
        // vips reports a 504-byte profile here and libviprs a 572-byte one;
        // the module docs say why the two generators cannot agree. What is
        // pinned is that a profile is always attached, as it is in vips.
        assert!(
            raster.icc_profile().is_some_and(|p| p.len() > 100),
            "jxlload always attaches the target-data profile"
        );
        // A still image carries no `n-pages` at all, as vips reports.
        assert_eq!(raster.get_field("n-pages"), None);
        assert_eq!(raster.get_n_pages(), 1);
        // The orientation is applied rather than tagged, so it reads
        // upright; vips writes nothing but 1 here either.
        assert_eq!(raster.orientation(), 1);
    }

    /**
     * Tests that an alpha channel survives the lossless decode as a fourth
     * band rather than being flattened, matching the `4 bands` vips reports
     * for the same file. Works by decoding the RGBA capture and comparing
     * both the colour ramp and the independent alpha ramp to the vips
     * values.
     * Input: `LOSSLESS_RGBA` -> Output: 4x3 `Rgba8` whose bytes equal the
     * source raster's.
     */
    #[cfg(feature = "jxl")]
    #[test]
    fn lossless_alpha_decodes_as_four_bands() {
        let raster = decode_bytes_with_limits(&LOSSLESS_RGBA, DecodeLimits::default())
            .expect("the vips lossless RGBA capture decodes");
        assert_eq!(raster.format(), PixelFormat::Rgba8);
        let expected: Vec<Vec<u8>> = RAMP_PIXELS
            .iter()
            .zip(RAMP_ALPHA)
            .map(|(rgb, a)| vec![rgb[0], rgb[1], rgb[2], a])
            .collect();
        assert_eq!(pixels(&raster), expected);
        assert_eq!(raster.data(), ramp_rgba().data(), "lossless is exact");
    }

    /**
     * Tests the place JPEG XL and WebP part company: a one-band image stays
     * one band, where `webpsave` promotes it to three because WebP stores
     * no greyscale. vips reports `1 band, b-w` for this capture. Works by
     * decoding the `b-w` capture and checking the format and the tag.
     * Input: `LOSSLESS_GREY` -> Output: 4x3 `Gray8` tagged `b-w`, pixels
     * equal to the luminance band of `RAMP_PIXELS`.
     */
    #[cfg(feature = "jxl")]
    #[test]
    fn greyscale_stays_one_band_unlike_webp() {
        let raster = decode_bytes_with_limits(&LOSSLESS_GREY, DecodeLimits::default())
            .expect("the vips greyscale capture decodes");
        assert_eq!(raster.format(), PixelFormat::Gray8);
        assert_eq!(raster.interpretation(), Interpretation::Bw);
        let expected: Vec<Vec<u8>> = RAMP_PIXELS.iter().map(|p| vec![p[0]]).collect();
        assert_eq!(pixels(&raster), expected);
    }

    /**
     * Tests that a 16-bit file decodes at its full range rather than being
     * narrowed to 8 bits on the way in, which is the failure mode a decoder
     * that guesses the carrier from the band count would have. vips reports
     * `ushort, rgb16` and `bits-per-sample: 16` for this capture. Works by
     * decoding it and comparing every sample to the `vips getpoint` values.
     * Input: `LOSSLESS_RGB16` -> Output: 4x3 `Rgb16` tagged `rgb16`,
     * samples equal to `RAMP16_PIXELS`, `bits-per-sample` 16.
     */
    #[cfg(feature = "jxl")]
    #[test]
    fn sixteen_bit_decodes_at_full_range() {
        let raster = decode_bytes_with_limits(&LOSSLESS_RGB16, DecodeLimits::default())
            .expect("the vips 16-bit capture decodes");
        assert_eq!(raster.format(), PixelFormat::Rgb16);
        assert_eq!(raster.interpretation(), Interpretation::Rgb16);
        assert_eq!(
            raster.get_field("bits-per-sample"),
            Some(MetadataValue::Int(16))
        );
        let got: Vec<Vec<u16>> = samples(&raster)
            .iter()
            .map(|p| p.iter().map(|s| *s as u16).collect())
            .collect();
        assert_eq!(got, RAMP16_PIXELS.map(Vec::from).to_vec());
        assert_eq!(raster.data(), ramp_rgb16().data(), "lossless is exact");
    }

    /**
     * Tests that a float file decodes as float rather than being quantised
     * to 8 or 16 bits, and lands on the linear-light tag vips gives it.
     * Every value in the fixture is a dyadic rational, so an exact
     * comparison is legitimate here and any rounding at all would show.
     * vips reports `float, scrgb` and `bits-per-sample: 32`. Works by
     * decoding the float capture and comparing every sample.
     * Input: `LOSSLESS_F32` -> Output: 4x3 `FloatF32(3)` tagged `scrgb`,
     * samples equal to the `vips getpoint` values, `bits-per-sample` 32.
     */
    #[cfg(feature = "jxl")]
    #[test]
    fn float_decodes_as_scrgb_without_quantising() {
        let raster = decode_bytes_with_limits(&LOSSLESS_F32, DecodeLimits::default())
            .expect("the vips float capture decodes");
        assert_eq!(
            raster.format(),
            PixelFormat::FloatF32(NonZeroU16::new(3).unwrap())
        );
        assert_eq!(raster.interpretation(), Interpretation::ScRgb);
        assert_eq!(
            raster.get_field("bits-per-sample"),
            Some(MetadataValue::Int(32))
        );
        let expected: Vec<Vec<f64>> = (0..3u32)
            .flat_map(|y| (0..4u32).map(move |x| (x, y)))
            .map(|(x, y)| {
                (0..3)
                    .map(|b| f64::from(x) * 0.25 + f64::from(y) * 0.0625 + f64::from(b) * 0.03125)
                    .collect()
            })
            .collect();
        for (got, want) in samples(&raster).iter().zip(&expected) {
            for (g, w) in got.iter().zip(want) {
                assert!(
                    (g - w).abs() < 1e-9,
                    "dyadic float samples survive exactly: got {g}, want {w}"
                );
            }
        }
    }

    /**
     * Tests the VarDCT path, which is the one place a tolerance is
     * legitimate: libjxl and jxl-oxide both implement a float inverse
     * transform, so they agree to within a count per channel rather than
     * byte for byte. The tolerance is the measured one and not a round
     * number: every channel of the capture lands within 1 of the vips
     * value, and seven of the thirty-six differ by exactly that. Works by
     * decoding the default-distance capture and comparing to the twelve
     * triples vips printed for the same file.
     * Input: `LOSSY_RGB` -> Output: 4x3 `Rgb8` within 1 of the vips values
     * everywhere, and not equal to the original ramp, because the encode
     * was lossy.
     */
    #[cfg(feature = "jxl")]
    #[test]
    fn vardct_decode_is_within_one_count_of_libjxl() {
        let raster = decode_bytes_with_limits(&LOSSY_RGB, DecodeLimits::default())
            .expect("the vips lossy capture decodes");
        assert_eq!(raster.format(), PixelFormat::Rgb8);
        let vips: [[u8; 3]; 12] = [
            [0, 0, 45],
            [80, 94, 23],
            [141, 196, 0],
            [172, 70, 0],
            [0, 149, 217],
            [150, 247, 206],
            [122, 97, 93],
            [213, 178, 0],
            [0, 51, 192],
            [51, 138, 208],
            [147, 239, 238],
            [221, 82, 208],
        ];
        let got = pixels(&raster);
        let mut differing = 0;
        for (p, (g, w)) in got.iter().zip(&vips).enumerate() {
            for (b, (gs, ws)) in g.iter().zip(w).enumerate() {
                let delta = i32::from(*gs) - i32::from(*ws);
                assert!(
                    delta.abs() <= 1,
                    "VarDCT parity is one count per channel, not {delta}, at pixel {p} band {b}"
                );
                differing += usize::from(delta != 0);
            }
        }
        assert_eq!(
            differing, 7,
            "the measured capture differs in exactly seven of thirty-six channels; \
             a change here means the decoder moved"
        );
        assert_ne!(got, RAMP_PIXELS.map(Vec::from).to_vec());
    }

    /**
     * Tests the multi-frame verdict: a three-frame JPEG XL loads its first
     * frame and says how many there were, which is exactly what a default
     * `vips jxlload` does (`n` defaults to 1). The toilet-roll load lives
     * in issue #621 behind the page model. Works by decoding a three-frame
     * capture and checking both the geometry and `n-pages`.
     * Input: `ANIM3` -> Output: 4x3 (not 4x9), pixels equal to frame 0,
     * `get_n_pages() == 3`.
     */
    #[cfg(feature = "jxl")]
    #[test]
    fn animated_jxl_loads_frame_zero_and_reports_the_page_count() {
        let raster = decode_bytes_with_limits(&ANIM3, DecodeLimits::default())
            .expect("the animation decodes");
        assert_eq!(
            (raster.width(), raster.height()),
            (4, 3),
            "one frame, not the 4x9 toilet roll"
        );
        assert_eq!(pixels(&raster), RAMP_PIXELS.map(Vec::from).to_vec());
        assert_eq!(raster.get_n_pages(), 3);
    }

    /**
     * Tests what the number under `n-pages` actually counts, which is the
     * question issue #635 was filed on: the frames in the original file and
     * nothing else. The test above cannot answer it, because `ANIM3` has
     * three frames, three bands and a height of three, so a loader that
     * attached the wrong one of those would still read as right. Works by
     * decoding a five-frame codestream, whose count collides with no other
     * number the raster carries.
     * Input: `ANIM5` -> Output: 4x3 with 3 bands, `get_n_pages() == 5`.
     */
    #[cfg(feature = "jxl")]
    #[test]
    fn n_pages_counts_the_frames_in_the_file_and_nothing_else() {
        let raster = decode_bytes_with_limits(&ANIM5, DecodeLimits::default())
            .expect("the five-frame codestream decodes");
        assert_eq!((raster.width(), raster.height()), (4, 3));
        assert_eq!(raster.bands(), 3);
        assert_eq!(
            raster.get_n_pages(),
            5,
            "n-pages is the frame count of the original, not the band count, \
             not an axis, and not the one page that was loaded"
        );
    }

    /**
     * Tests the EXIF fix-up, which is the one metadata transform on this
     * path: a JPEG XL `Exif` box holds a big-endian 4-byte offset and no
     * `Exif\0\0` prefix, so the loader has to skip the first and restore
     * the second before the blob matches what the JPEG loader attaches.
     * Two fixtures carry the same TIFF block at offsets 0 and 6, and vips
     * reports the identical 16-byte blob for both, which is what proves the
     * offset is skipped rather than ignored. The `xml ` box needs no
     * transform at all and comes back verbatim.
     * Input: `META_OFF0` and `META_OFF6` -> Output: `exif-data` equal to
     * `EXIF_BLOB` from both, `xmp-data` equal to `XMP_PACKET` from the
     * first, and the ramp pixels unchanged in both.
     */
    #[cfg(feature = "jxl")]
    #[test]
    fn exif_box_gets_the_prefix_back_and_the_offset_skipped() {
        for (name, bytes) in [
            ("tiff_header_offset 0", &META_OFF0[..]),
            ("tiff_header_offset 6", &META_OFF6[..]),
        ] {
            let raster = decode_bytes_with_limits(bytes, DecodeLimits::default())
                .unwrap_or_else(|e| panic!("{name} decodes: {e}"));
            assert_eq!(
                pixels(&raster),
                RAMP_PIXELS.map(Vec::from).to_vec(),
                "{name}"
            );
            assert_eq!(blob(&raster, "exif-data"), EXIF_BLOB, "{name}");
        }
        let raster = decode_bytes_with_limits(&META_OFF0, DecodeLimits::default()).unwrap();
        assert_eq!(blob(&raster, "xmp-data"), XMP_PACKET);
        // The offset-6 fixture carries no `xml ` box, so nothing is
        // invented for it.
        let bare = decode_bytes_with_limits(&META_OFF6, DecodeLimits::default()).unwrap();
        assert_eq!(bare.get_field("xmp-data"), None);
    }

    /**
     * Tests the deliberate divergence: an `Exif` box whose offset runs past
     * its payload costs the blob and not the image. vips takes the other
     * branch and fails the whole load, warning `invalid data in EXIF box`
     * (measured: `vipsheader` exits 1 and prints no header), which is the
     * wrong trade for a decoder reading untrusted bytes. Works by decoding
     * the malformed capture and checking the pixels arrived and the field
     * did not.
     * Input: `META_BAD_OFFSET` -> Output: 4x3 `Rgb8` with the ramp pixels
     * and no `exif-data` field.
     */
    #[cfg(feature = "jxl")]
    #[test]
    fn a_malformed_exif_box_costs_the_blob_and_not_the_image() {
        let raster = decode_bytes_with_limits(&META_BAD_OFFSET, DecodeLimits::default())
            .expect("the pixels are fine even though the EXIF box is not");
        assert_eq!(pixels(&raster), RAMP_PIXELS.map(Vec::from).to_vec());
        assert_eq!(raster.get_field("exif-data"), None);
    }

    /**
     * Tests the property the lossless-only encoder buys: because there is
     * no quantisation step anywhere in the pipeline, decoding what
     * `encode_jxl` wrote returns the input bytes exactly, for every carrier
     * the encoder accepts. That includes 16-bit, which is where JPEG XL and
     * WebP part company: `encode_webp` refuses a wide raster because the
     * format has no 16-bit sample, and this one does not have to.
     * Works by encoding five rasters at the default options and decoding
     * each result back.
     * Input: the 4x3 `Rgb8`, `Rgba8`, `Gray8`, `Multi8(2)` and `Rgb16`
     * ramps -> Output: identical dimensions, identical pixel format,
     * byte-identical data.
     */
    #[cfg(feature = "jxl")]
    #[test]
    fn lossless_encode_decode_is_the_identity() {
        let two = NonZeroU16::new(2).unwrap();
        let grey = Raster::new(
            4,
            3,
            PixelFormat::Gray8,
            RAMP_PIXELS.iter().map(|p| p[0]).collect::<Vec<u8>>(),
        )
        .unwrap();
        let grey_alpha = Raster::new(
            4,
            3,
            PixelFormat::Multi8(two),
            RAMP_PIXELS
                .iter()
                .zip(RAMP_ALPHA)
                .flat_map(|(p, a)| [p[0], a])
                .collect::<Vec<u8>>(),
        )
        .unwrap();
        for original in [ramp_rgb(), ramp_rgba(), grey, grey_alpha, ramp_rgb16()] {
            let bytes = original
                .encode_jxl(SaveOptions::default())
                .unwrap_or_else(|e| panic!("{:?} encodes: {e}", original.format()));
            assert_eq!(
                &bytes[..2],
                b"\xff\x0a",
                "the encoder writes a bare codestream, as vips does at --keep none"
            );
            let back = decode_bytes_with_limits(&bytes, DecodeLimits::default())
                .unwrap_or_else(|e| panic!("our own {:?} bytes decode: {e}", original.format()));
            assert_eq!((back.width(), back.height()), (4, 3));
            assert_eq!(
                back.format(),
                original.format(),
                "the carrier survives the round trip"
            );
            assert_eq!(back.data(), original.data(), "lossless is exact");
        }
    }

    /**
     * Tests that a float raster is refused with a message naming the remedy
     * rather than silently quantised. vips writes float samples natively
     * and libviprs cannot, because `zune-jpegxl` holds 8- and 16-bit
     * integers only (`errors.rs:SUPPORTED_DEPTHS`), so the caller picks the
     * quantisation. Works by encoding an `RgbaF32` raster and a
     * `FloatF32(3)` one and matching the typed error.
     * Input: 4x3 float rasters -> Output: `EncodeError::Encode` naming the
     * format and saying to cast.
     */
    #[cfg(feature = "jxl")]
    #[test]
    fn float_is_refused_rather_than_quantised() {
        for format in [
            PixelFormat::RgbaF32,
            PixelFormat::FloatF32(NonZeroU16::new(3).unwrap()),
        ] {
            let wide = Raster::zeroed(4, 3, format).unwrap();
            let err = wide
                .encode_jxl(SaveOptions::default())
                .expect_err("zune-jpegxl has no float depth");
            assert!(matches!(err, EncodeError::Encode(_)), "{err:?}");
            let msg = err.to_string();
            assert!(msg.contains("float"), "{msg}");
            assert!(msg.contains("cast"), "{msg}");
        }
    }

    /**
     * Tests the floor libviprs has and vips does not: `zune-jpegxl` refuses
     * a single-pixel row or column outright, where `vips jxlsave` writes an
     * 18-byte 1x1 file happily (measured: 1x1, 2x1, 1x2 and 4x1 all round
     * trip through vips). The refusal names the floor rather than letting
     * the dependency's own `ZeroDimension("width")` reach the caller. Works
     * by encoding at each geometry either side of the floor.
     * Input: 1x3, 3x1 and 2x2 `Rgb8` -> Output: an `EncodeError::Encode`
     * naming the geometry and the floor for the first two, and bytes for
     * the third.
     */
    #[cfg(feature = "jxl")]
    #[test]
    fn a_one_pixel_axis_is_refused_with_the_floor_named() {
        for (w, h) in [(1u32, 3u32), (3, 1)] {
            let thin = Raster::zeroed(w, h, PixelFormat::Rgb8).unwrap();
            let err = thin
                .encode_jxl(SaveOptions::default())
                .expect_err("zune-jpegxl refuses a single-pixel axis");
            assert!(matches!(err, EncodeError::Encode(_)), "{err:?}");
            let msg = err.to_string();
            assert!(msg.contains(&format!("{w}x{h}")), "{msg}");
            assert!(msg.contains("2 pixels on each axis"), "{msg}");
        }
        let ok = Raster::zeroed(MIN_DIMENSION, MIN_DIMENSION, PixelFormat::Rgb8).unwrap();
        assert!(ok.encode_jxl(SaveOptions::default()).is_ok());
    }

    /**
     * Tests that the decode limits reach this decoder rather than stopping
     * at the `image` facade, and that they are applied to the DECLARED
     * header geometry rather than after the frame is built: the decode is
     * fed in two phases exactly so the ceilings run before the pixel
     * buffer is reserved. Works by decoding the animation capture under a
     * coordinate ceiling, a pixel ceiling and an allocation ceiling all
     * below its 4x3 geometry, and checking each reports its own typed
     * variant.
     * Input: `ANIM3` under `max_coord = 2`, under `max_pixels = 4`, and
     * under `max_alloc_bytes = 8` -> Output: `CoordLimitExceeded` and
     * `DimensionLimitExceeded` naming 4x3, and a `JxlError` allocation
     * variant carrying the budget.
     */
    #[cfg(feature = "jxl")]
    #[test]
    fn decode_limits_are_enforced_on_the_declared_geometry() {
        let tight = DecodeLimits::default().with_max_coord(2);
        assert!(matches!(
            decode_jxl(&ANIM3, tight),
            Err(SourceError::CoordLimitExceeded {
                width: 4,
                height: 3,
                max_coord: 2
            })
        ));
        let small = DecodeLimits::default().with_max_pixels(4);
        assert!(matches!(
            decode_jxl(&ANIM3, small),
            Err(SourceError::DimensionLimitExceeded {
                width: 4,
                height: 3,
                max_pixels: 4
            })
        ));
        // The allocation budget is separate: 12 pixels are inside every
        // pixel ceiling above and still need 36 bytes of frame buffer.
        // Which of the two allocation variants answers is not a detail to
        // wave at, and on this file it is the decoder's tracker rather
        // than the crate's pre-check: the same budget is handed to
        // `jxl-oxide`'s `AllocTracker`, which is live from the first
        // `feed_bytes`, and 8 bytes is under even the working buffers a
        // 4x3 header needs. The pre-check answers instead once the
        // declared frame is large next to that working set, which
        // `the_declared_frame_is_priced_before_the_frame_data_is_fed_in`
        // pins. Measured, not assumed: the two swap over between the two
        // tests, and an assertion that took either would pass whichever
        // way round it went.
        let starved = DecodeLimits::default().with_max_alloc_bytes(8);
        let err = decode_jxl(&ANIM3, starved).expect_err("8 bytes is not a 4x3 RGB frame");
        assert!(
            matches!(
                err,
                SourceError::Jxl(JxlError::DecoderAllocLimitExceeded { max_alloc_bytes: 8 })
            ),
            "{err:?}"
        );
    }

    /**
     * Tests the crate's own allocation pre-check, the one priced from the
     * declared header geometry before a byte of frame data is fed in, and
     * pins it as a variant distinct from the decoder's own tracker. The
     * two split on the ratio between the declared frame and `jxl-oxide`'s
     * working set rather than on the budget alone: measured on this
     * decoder, a 4x3 file answers `DecoderAllocLimitExceeded` at every
     * budget under its 36-byte frame, while a 512x512 one answers
     * `AllocLimitExceeded` at every budget under its 786432-byte frame,
     * because the header-phase working buffers are small next to it.
     * Without a case on this side of that line the pre-check would be
     * unreachable code that no test would notice losing.
     * Input: a 512x512 `Rgb8` raster round-tripped through the encoder,
     * decoded under `max_alloc_bytes = 256 KiB` -> Output:
     * `JxlError::AllocLimitExceeded` naming 512x512x3 and 786432 bytes
     * against the 262144-byte budget, and a clean decode once the budget
     * clears the frame.
     */
    #[cfg(feature = "jxl")]
    #[test]
    fn the_declared_frame_is_priced_before_the_frame_data_is_fed_in() {
        let big = Raster::zeroed(512, 512, PixelFormat::Rgb8).unwrap();
        let bytes = big.encode_jxl(SaveOptions::default()).unwrap();
        let needed = 512u64 * 512 * 3;

        let starved = DecodeLimits::default().with_max_alloc_bytes(256 * 1024);
        let err = decode_jxl(&bytes, starved).expect_err("256 KiB is not a 512x512 RGB frame");
        assert!(
            matches!(
                err,
                SourceError::Jxl(JxlError::AllocLimitExceeded {
                    width: 512,
                    height: 512,
                    channels: 3,
                    needed: n,
                    max_alloc_bytes: 262_144,
                }) if n == needed
            ),
            "{err:?}"
        );
        // The refusal names the knob rather than leaving the caller to
        // guess which of the ceilings bit.
        let message = err.to_string();
        assert!(message.contains("786432"), "{message}");
        assert!(message.contains("262144"), "{message}");

        // And the same file decodes once the budget covers the frame, so
        // the refusal is the budget and not the bytes.
        let roomy = DecodeLimits::default().with_max_alloc_bytes(4 * 1024 * 1024);
        let back = decode_jxl(&bytes, roomy).expect("a 4 MiB budget holds a 512x512 RGB frame");
        assert_eq!((back.width(), back.height()), (512, 512));
    }

    /**
     * Tests that the frame pre-check bites at exactly the byte the declared
     * frame costs, and not one byte either side. The case above refuses at
     * 256 KiB against a 768 KiB frame, which a price wrong by a factor
     * would also refuse; only the exact pair below pins the arithmetic and
     * the `>` in the comparison.
     * One byte short of the frame the pre-check answers and names the
     * price. At exactly the frame it does not, and that is what proves the
     * comparison is `>` and not `>=`: the refusal that arrives instead is
     * `jxl-oxide`'s own tracker, which holds the same budget and is still
     * carrying its header-phase buffers when the frame is asked for. So
     * the accepting half of this boundary is not reachable through the
     * decoder, and asserting a clean decode there would be asserting the
     * tracker's ceiling rather than this one.
     * Input: the same 512x512 `Rgb8` file at `max_alloc_bytes` 786431 then
     * 786432 -> Output: `AllocLimitExceeded { needed: 786432 }`, then
     * `DecoderAllocLimitExceeded`.
     */
    #[cfg(feature = "jxl")]
    #[test]
    fn the_frame_budget_bites_at_exactly_the_declared_price() {
        let big = Raster::zeroed(512, 512, PixelFormat::Rgb8).unwrap();
        let bytes = big.encode_jxl(SaveOptions::default()).unwrap();
        let price = 512u64 * 512 * 3;

        let short = DecodeLimits::default().with_max_alloc_bytes(price - 1);
        let err = decode_jxl(&bytes, short).expect_err("786431 bytes is one short of the frame");
        assert!(
            matches!(
                err,
                SourceError::Jxl(JxlError::AllocLimitExceeded {
                    width: 512,
                    height: 512,
                    channels: 3,
                    needed: 786_432,
                    max_alloc_bytes: 786_431,
                })
            ),
            "{err:?}"
        );

        let exact = DecodeLimits::default().with_max_alloc_bytes(price);
        let err = decode_jxl(&bytes, exact).expect_err("the decoder needs room of its own too");
        assert!(
            matches!(
                err,
                SourceError::Jxl(JxlError::DecoderAllocLimitExceeded {
                    max_alloc_bytes: 786_432
                })
            ),
            "a frame priced at exactly the budget must clear the pre-check, so the \
             only refusal left is the decoder's own tracker: {err:?}"
        );

        // And again on a carrier wider than a byte, because an 8-bit frame
        // cannot tell the sample size apart from 1: dropping
        // `bytes_per_channel()` from the price leaves the 512x512 `Rgb8`
        // case above answering exactly the same thing. A 256x256 `Rgb16`
        // frame is 393216 bytes and only half that without it.
        let deep = Raster::zeroed(256, 256, PixelFormat::Rgb16).unwrap();
        let deep_bytes = deep.encode_jxl(SaveOptions::default()).unwrap();
        let deep_price = 256u64 * 256 * 3 * 2;
        let short = DecodeLimits::default().with_max_alloc_bytes(deep_price - 1);
        let err = decode_jxl(&deep_bytes, short).expect_err("393215 bytes is one short");
        assert!(
            matches!(
                err,
                SourceError::Jxl(JxlError::AllocLimitExceeded {
                    width: 256,
                    height: 256,
                    channels: 3,
                    needed: 393_216,
                    max_alloc_bytes: 393_215,
                })
            ),
            "{err:?}"
        );
    }

    /**
     * Tests that a truncated or non-JPEG-XL buffer is refused with a typed
     * error rather than a panic or a zero-sized raster, since these bytes
     * are the untrusted end of the crate. The two-phase feed makes the
     * header truncation and the frame truncation different code paths, so
     * both are exercised. Works by feeding the decoder prefixes of both
     * container forms, a signature with no codestream behind it, and an
     * empty buffer.
     * Input: five malformed buffers -> Output: the `JxlError` variant
     * named beside each, and never a raster. Each expectation is the
     * variant rather than `is_err`, because `Decode` and `Truncated` are
     * different answers to different questions and a test that took
     * either would not notice one turning into the other.
     */
    #[cfg(feature = "jxl")]
    #[test]
    fn malformed_input_is_refused_with_a_typed_error() {
        // `Truncated` is the answer when `jxl-oxide` asks for more bytes
        // rather than refusing what it has, which is what a prefix of a
        // well-formed file produces; `Decode` is the answer when the bytes
        // it has are wrong.
        for (name, bytes, expect_truncated) in [
            (
                "truncated in the codestream header",
                &LOSSLESS_RGB[..6],
                true,
            ),
            ("truncated in the frame body", &LOSSLESS_RGB[..40], true),
            ("container signature only", &META_OFF0[..12], true),
            ("container with no codestream", &META_OFF0[..32], true),
            ("empty", &[][..], true),
        ] {
            let err = decode_jxl(bytes, DecodeLimits::default())
                .expect_err(&format!("{name} should not decode"));
            match err {
                SourceError::Jxl(JxlError::Truncated { expected }) => {
                    assert!(expect_truncated, "{name} reported Truncated: {expected}");
                    assert!(
                        expected == "the end of the image header"
                            || expected == "the first complete frame",
                        "{name} named {expected}"
                    );
                }
                SourceError::Jxl(JxlError::Decode { ref message }) => {
                    assert!(!expect_truncated, "{name} reported Decode: {message}");
                }
                other => panic!("{name} should be a typed JxlError, got {other:?}"),
            }
        }
    }

    /**
     * Tests the CMYK refusal, which is the one colour space this loader
     * turns away, through the seam it is checked at. It cannot be reached
     * from a file: nothing in this build writes a CMYK JPEG XL, because
     * `vips jxlsave` converts a `cmyk` image to sRGB on the way in, so the
     * check would otherwise be untested code guarding an untestable case.
     * Works by handing `check_colour_space` every `jxl_oxide::PixelFormat`
     * there is and asserting which ones carry a black ink channel.
     * Input: the six `jxl_oxide::PixelFormat` values -> Output: `Ok` for
     * the four without black, `JxlError::CmykNotSupported` naming the
     * space for `Cmyk` and `Cmyka`.
     */
    #[cfg(feature = "jxl")]
    #[test]
    fn a_black_ink_channel_is_refused_by_name() {
        use jxl_oxide::PixelFormat as JxlPixelFormat;

        for ok in [
            JxlPixelFormat::Gray,
            JxlPixelFormat::Graya,
            JxlPixelFormat::Rgb,
            JxlPixelFormat::Rgba,
        ] {
            assert!(check_colour_space(ok).is_ok(), "{ok:?} has no black ink");
        }

        for (space, name) in [
            (JxlPixelFormat::Cmyk, "Cmyk"),
            (JxlPixelFormat::Cmyka, "Cmyka"),
        ] {
            let err = check_colour_space(space).expect_err("a black channel is refused");
            assert!(
                matches!(&err, JxlError::CmykNotSupported { pixel_format } if pixel_format == name),
                "{err:?}"
            );
            // The message says why, not just that: both routes out are
            // named so a caller knows this is a wiring gap rather than a
            // capability the crate lacks.
            let message = err.to_string();
            assert!(message.contains("ColorManagementSystem"), "{message}");
            assert!(message.contains("NullCms"), "{message}");
            assert!(message.contains("crate::colour"), "{message}");
        }
    }

    /**
     * Tests the two defensive refusals no file can reach, so they are
     * pinned by something rather than by nothing: a channel count with no
     * raster carrier, and a rendered frame that disagrees with the header
     * the allocation budget was priced against. `jxl_oxide::PixelFormat`
     * names at most five channels and renders what the header declared, so
     * neither is reachable through `decode_jxl` today; both exist because
     * the alternative to the check is a panic or a buffer sized for the
     * wrong image.
     * Input: 0 and 65536 channels, and a 3-against-4 channel disagreement
     * -> Output: `JxlError::UnsupportedChannelCount` naming the count and
     * the 65535-band ceiling, and `JxlError::ChannelCountMismatch` naming
     * both sides.
     */
    #[cfg(feature = "jxl")]
    #[test]
    fn the_defensive_channel_checks_report_their_own_variants() {
        for bands in [0u32, u32::from(u16::MAX) + 1] {
            let err = carrier(bands, false, 8)
                .err()
                .unwrap_or_else(|| panic!("{bands} bands should have no raster carrier"));
            assert!(
                matches!(
                    err,
                    JxlError::UnsupportedChannelCount {
                        channels,
                        max: 65535
                    } if channels == bands
                ),
                "{err:?}"
            );
        }
        // Every count `jxl-oxide` can actually report does have a carrier.
        for bands in 1..=5u32 {
            assert!(carrier(bands, false, 8).is_ok(), "{bands} bands");
        }

        assert!(check_channel_count(3, 3).is_ok());
        let err = check_channel_count(3, 4).expect_err("3 declared is not 4 rendered");
        assert!(
            matches!(
                err,
                JxlError::ChannelCountMismatch {
                    declared: 3,
                    rendered: 4
                }
            ),
            "{err:?}"
        );
    }

    /**
     * Tests that both container forms reach the same decoder through the
     * shared sniff route, because JPEG XL is the only format in the crate
     * with two unrelated magics: the bare codestream starts `FF 0A` and the
     * ISOBMFF form starts with a 12-byte signature box. A sniff table that
     * knew only one of them would silently drop half the format.
     * Input: `LOSSLESS_RGB` (bare) and `LOSSLESS_RGB16` (container) ->
     * Output: both decode to 4x3 rasters through `decode_bytes_with_limits`,
     * with the carrier each file declares.
     */
    #[cfg(feature = "jxl")]
    #[test]
    fn both_container_forms_reach_the_decoder_through_the_sniff_route() {
        assert_eq!(&LOSSLESS_RGB[..2], b"\xff\x0a");
        assert_eq!(
            &LOSSLESS_RGB16[..12],
            b"\x00\x00\x00\x0cJXL \x0d\x0a\x87\x0a"
        );
        let bare = decode_bytes_with_limits(&LOSSLESS_RGB, DecodeLimits::default()).unwrap();
        let boxed = decode_bytes_with_limits(&LOSSLESS_RGB16, DecodeLimits::default()).unwrap();
        assert_eq!(bare.format(), PixelFormat::Rgb8);
        assert_eq!(boxed.format(), PixelFormat::Rgb16);
        assert_eq!((boxed.width(), boxed.height()), (4, 3));
    }

    /**
     * Pins the shape of the options struct: the one field defaults to the
     * mode with an encoder behind it, and the struct is open enough to
     * build with `..Default::default()` from outside its own module. There
     * is deliberately no `keep` field, because the encoder writes no boxes
     * at all.
     * Input: none -> Output: every spelling compares equal, with
     * compression `Lossless`.
     */
    #[test]
    fn save_options_default_is_lossless_and_updatable() {
        let explicit = SaveOptions {
            compression: Compression::Lossless,
        };
        let updated = SaveOptions {
            ..Default::default()
        };
        assert_eq!(SaveOptions::default(), explicit);
        assert_eq!(updated, explicit);
        assert_eq!(Compression::default(), Compression::Lossless);
    }

    /**
     * Tests that a build without the `jxl` feature still exposes every
     * entry point at its real signature and still reports a typed error
     * naming the capability, so a caller compiled either way gets one
     * shape and can match on the failure instead of hitting a missing
     * symbol. It also pins which error each entry point reports, because
     * they are deliberately not the same one: the decoder reports
     * `JxlError::FeatureNotEnabled` through `SourceError::Jxl`, the one
     * variant of that enum a feature-on build never produces, and the
     * encoders report the `EncodeError::Unsupported` every format with no
     * encoder in this build reports, so a caller matching on the encode
     * spine does not have to learn a second variant.
     * Input: the 71-byte `LOSSLESS_RGB` capture and the 4x3 ramp ->
     * Output: `SourceError::Jxl(JxlError::FeatureNotEnabled)` naming JPEG
     * XL and the feature, `EncodeError::Unsupported { format: "jxl" }`,
     * and a `SaveError` carrying the same wording.
     */
    #[test]
    #[cfg(not(feature = "jxl"))]
    fn without_the_feature_every_entry_point_is_a_typed_refusal() {
        let err = decode_jxl(&LOSSLESS_RGB, DecodeLimits::default()).unwrap_err();
        assert!(
            matches!(err, SourceError::Jxl(JxlError::FeatureNotEnabled)),
            "expected the feature-off variant, got {err:?}"
        );
        assert!(
            err.to_string().contains("JPEG XL") && err.to_string().contains("`jxl`"),
            "the refusal names the format and the feature, got {err}"
        );

        let raster = ramp_rgb();
        let err = raster.encode_jxl(SaveOptions::default()).unwrap_err();
        assert!(
            matches!(err, EncodeError::Unsupported { ref format } if format == "jxl"),
            "{err:?}"
        );

        let dir = tempfile::tempdir().unwrap();
        let err = raster
            .save_jxl(&dir.path().join("out.jxl"), SaveOptions::default())
            .unwrap_err();
        assert!(
            err.to_string().contains("jxl"),
            "save_jxl carries the encoder's wording, got {err}"
        );
    }

    /**
     * Sweeps the seeded fuzz corpus through the decoder, so every
     * malformation it holds is a `cargo test` regression rather than
     * something only a fuzz run would notice. The naming carries the
     * assertion, in three grades: a `valid-` seed must decode (which
     * includes the two `Exif` shapes libviprs deliberately tolerates and
     * vips does not), a `rejected-` seed must come back as a typed error,
     * and a `nocrash-` seed is only required not to panic, because a
     * single-bit flip can land on a file that is still perfectly legal and
     * pinning a direction for it would pin a coincidence. Between them the
     * seeds cover both container forms, the VarDCT path, a three-frame
     * file, boxes whose declared length overruns or underruns the file,
     * and flips through the header bytes where the geometry and the
     * channel counts live.
     * Input: every file under `fuzz/corpus/fuzz_jxl/` -> Output: a raster
     * from each `valid-` seed, an `Err` from each `rejected-` one, and no
     * panic from any of them.
     */
    #[cfg(feature = "jxl")]
    #[test]
    fn the_fuzz_corpus_decodes_or_fails_exactly_as_named() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fuzz")
            .join("corpus")
            .join("fuzz_jxl");
        // The same budget `fuzz_targets/fuzz_jxl.rs` runs under, so a seed
        // that trips the ceiling there trips it here too.
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
            let result = decode_jxl(&bytes, limits);
            seen += 1;
            if name.starts_with("valid-") {
                assert!(result.is_ok(), "{name} should decode: {:?}", result.err());
            } else if name.starts_with("rejected-") {
                // The variant, not `is_err`: every seeded rejection is a
                // JPEG XL refusal and must arrive on `JxlError`, so a
                // seed that started failing for some unrelated reason
                // (an I/O error, a shared ceiling) shows up here rather
                // than passing quietly.
                let err = result.expect_err(&format!("{name} should not decode"));
                assert!(
                    matches!(err, SourceError::Jxl(_)),
                    "{name} should be a typed JxlError, got {err:?}"
                );
            } else {
                assert!(
                    name.starts_with("nocrash-"),
                    "{name} needs one of the three outcome prefixes"
                );
            }
        }
        assert!(seen >= 26, "the corpus lost seeds: only {seen} left");
    }

    /// Write the files the `--viprs` half of
    /// `oracle-captures/foreign-jxl/capture.py` reads back through vips.
    ///
    /// Ignored because it writes outside the target directory and needs
    /// `JXL_ORACLE_OUT` to say where; it is a capture step rather than an
    /// assertion. Run it as
    /// `JXL_ORACLE_OUT=oracle-captures/foreign-jxl/outputs cargo test
    /// --lib jxl::tests::write_oracle_inputs -- --ignored`.
    #[cfg(feature = "jxl")]
    #[test]
    #[ignore = "capture step: writes fixtures for capture.py --viprs, needs JXL_ORACLE_OUT"]
    fn write_oracle_inputs() {
        let dir =
            std::env::var("JXL_ORACLE_OUT").expect("JXL_ORACLE_OUT names an output directory");
        let grey = Raster::new(
            4,
            3,
            PixelFormat::Gray8,
            RAMP_PIXELS.iter().map(|p| p[0]).collect::<Vec<u8>>(),
        )
        .unwrap();
        for (name, raster) in [
            ("viprs_rgb", ramp_rgb()),
            ("viprs_rgba", ramp_rgba()),
            ("viprs_grey", grey),
            ("viprs_rgb16", ramp_rgb16()),
        ] {
            raster
                .save_jxl(
                    Path::new(&format!("{dir}/{name}.jxl")),
                    SaveOptions::default(),
                )
                .unwrap();
        }
    }
}
