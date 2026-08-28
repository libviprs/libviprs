//! OpenEXR (`.exr`) load: scene-linear samples in, float bands out.
//!
//! Ported from libvips `foreign/openexrload.c` and
//! `foreign/openexr2vips.c`, which drive the OpenEXR C++ library through
//! its **C RGBA wrapper** (`ImfCRgbaFile.h`). That wrapper is the whole
//! story of this module's divergences: it hands back four `half` samples
//! per pixel and nothing else, so vips flattens every EXR to RGBA half
//! before it ever sees a float. `openexr2vips.c:24-38` says so in its own
//! TODO list, "more of OpenEXR's pixel formats", "more than just RGBA
//! channels", "best redo with the C++ API now we support C++ operations".
//!
//! There is no save side, here or upstream. `vips -l` registers
//! `VipsForeignLoadOpenexr (openexrload)` and no saver at all, and
//! `vips copy src.png out.exr` answers `"out.exr" is not a known file
//! format`. Nothing is deferred; there is nothing to be parity with.
//!
//! # Operations
//!
//! | libviprs method | libvips equivalent | result |
//! |---|---|---|
//! | [`decode_exr`] | `openexrload` | float raster with one band per selected channel ([`PixelFormat::RgbaF32`] at four bands, [`PixelFormat::FloatF32`]`(n)` otherwise), tagged [`Interpretation::ScRgb`] for an RGB selection and [`Interpretation::Multiband`] otherwise |
//! | *(none)* | *(none)* | libvips has never shipped an EXR writer |
//!
//! # Semantics
//!
//! * **The carrier keeps the file's precision, and vips does not.** vips
//!   reads through `ImfRgba`, four `half` samples, and widens them with
//!   `ImfHalfToFloatArray` (`openexr2vips.c:341-347`, `:411-412`). A HALF
//!   file therefore comes back exact, and a **FLOAT file comes back
//!   rounded to half**. Measured on `rgba_float_fine.exr`, whose green
//!   channel holds `7/3`: vips reports `2.333984375`, which is
//!   `f16::from_f32(2.3333333)` widened again, where the file holds
//!   `2.3333333`. libviprs returns the stored `f32`. Same for UINT, which
//!   vips also funnels through half and so saturates to infinity above
//!   65504.
//! * **Band count follows the file.** vips always emits four bands
//!   (`openexr2vips.c:222-224` passes a literal `4`), filling a missing
//!   alpha with `1.0` and a missing colour channel with `0.0`. Measured:
//!   an R/G/B file reads back as `(r, g, b, 1.0)`, a `Y`-only file as
//!   `(y, y, y, 1.0)`, and a `Z`-only file as **`(0, 0, 0, 1)`, an
//!   entirely black image with no error**. libviprs selects channels by
//!   name instead and emits exactly what it selected; see [`decode_exr`]
//!   for the rule. A depth pass survives here and does not survive vips.
//! * **Geometry comes from the data window, and the display window is
//!   ignored.** `read_new` (`openexr2vips.c:186-191`) sizes the image
//!   `xmax - xmin + 1` by `ymax - ymin + 1` and the frame-buffer base
//!   pointer is backed off by the window origin (`:302-305`, `:401-403`),
//!   so a data window at `(5, 7)` decodes to the same pixels at `(0, 0)`.
//!   Both measured. libviprs matches, and records the origin it dropped in
//!   `exr-data-window-left` / `exr-data-window-top`, which vips discards
//!   entirely.
//! * **First part only.** vips opens with `ImfOpenTiledInputFile` and then
//!   `ImfOpenInputFile` (`openexr2vips.c:164-165`); neither is multi-part
//!   aware, so parts past the first are unreachable there. libviprs reads
//!   the first part too and reports the part count as `exr-parts` so the
//!   caller can at least see what was skipped. Deliberately **not**
//!   `n-pages`: `vipsheader -a` reports no such field for any EXR, an EXR
//!   part is a layer rather than a page, and [`decode_exr`] has no part
//!   selector, so putting the count behind [`Raster::get_n_pages`] would
//!   invite a `0..n` sweep over parts nothing here can address.
//! * **Tiled and scanline files both decode, and tiled ones carry their
//!   tile geometry.** `read_header` (`openexr2vips.c:230-232`) sets
//!   `VIPS_META_TILE_WIDTH` / `VIPS_META_TILE_HEIGHT` for a tiled file and
//!   nothing for a scanline one; libviprs sets `tile-width` /
//!   `tile-height` on the same condition.
//! * **Compression is the decoder's business, not this module's.** NONE,
//!   RLE, ZIPS, ZIP and PIZ are lossless, and PXR24 is lossless for HALF
//!   samples (it only truncates `f32` to 24 bits). All six were measured
//!   decoding to byte-identical payloads through vips, which is why the
//!   parity pins in this module are **exact equality with no tolerance**.
//!   B44 and B44A are lossy by construction and DWAA/DWAB are lossy in
//!   general, so those four are pinned against their own captured bytes
//!   rather than against the lossless payload. All four turn out to agree
//!   with the OpenEXR C++ decoders bit for bit anyway.
//! * **UINT channels are refused rather than mangled.** libviprs has no
//!   unsigned-integer sample carrier yet (issue #517), and widening a
//!   32-bit object ID to `f32` loses every value above 2^24. vips does not
//!   refuse: it converts UINT to half, so an ID of 100000 reads back as
//!   infinity. [`ExrError::UnsupportedSampleType`] is the honest answer
//!   until #517 lands.
//! * **Deep EXR is refused.** vips cannot read it either
//!   (`ImfOpenInputFile` fails on a deep file), it just reports the
//!   OpenEXR message rather than a typed error.
//!
//! Every number this module is pinned against was measured on the real
//! vips 8.18.4 binary against fixtures written by the OpenEXR reference
//! implementation 3.4.15, and both are recorded with the commands that
//! produced them in `oracle-captures/foreign-exr/`.
//!
//! [`decode_exr`] is fallible and has no panicking twin, matching the rest
//! of the codec surface in [`crate::radiance`], [`crate::webp`] and
//! [`crate::gif`]: a decoder's failures come from untrusted bytes, so a
//! panicking spelling would have no honest caller.

use std::io::Cursor;
use std::num::NonZeroU16;

use exr::prelude::{ReadChannels, ReadLayers};
use thiserror::Error;

use crate::conversion::Interpretation;
use crate::imageio::MetadataValue;
use crate::pixel::PixelFormat;
use crate::raster::{Raster, RasterError, buffer_len};
use crate::source::{DecodeLimits, SourceError};

/// The four magic bytes every OpenEXR file opens with
/// (`vips__openexr_isexr`, `openexr2vips.c:105-115`).
pub(crate) const MAGIC: [u8; 4] = [0x76, 0x2f, 0x31, 0x01];

/// Bytes per decoded sample. The carrier is always `f32`.
const SAMPLE_BYTES: usize = 4;

/// Name of the OpenEXR red channel.
const CHANNEL_R: &str = "R";
/// Name of the OpenEXR green channel.
const CHANNEL_G: &str = "G";
/// Name of the OpenEXR blue channel.
const CHANNEL_B: &str = "B";
/// Name of the OpenEXR alpha channel.
const CHANNEL_A: &str = "A";
/// Name of the OpenEXR luminance channel.
const CHANNEL_Y: &str = "Y";

/// Errors from the OpenEXR loader.
///
/// Every variant except [`ExrError::Raster`] and [`ExrError::Decode`]
/// describes a specific thing this build will not carry, which is what
/// makes them worth typing: the fuzz corpus in `fuzz/corpus/fuzz_exr/`
/// asserts on the variant, not on a message.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ExrError {
    /// The first four bytes are not `76 2F 31 01`.
    ///
    /// vips checks the same four bytes in `vips__openexr_isexr`
    /// (`openexr2vips.c:105-115`).
    #[error("exr: expected the magic bytes 76 2f 31 01, found {found:02x?}")]
    BadMagic {
        /// The leading bytes as read, up to four of them.
        found: Vec<u8>,
    },
    /// The `exr` crate refused the file, or the bytes ran out inside it.
    ///
    /// This is the untyped tail: the reference format has a large surface
    /// of malformations (bad attribute types, impossible chunk tables,
    /// truncated compressed blocks) and reproducing that taxonomy here
    /// would be a second parser. The message is the decoder's own.
    #[error("exr: {message}")]
    Decode {
        /// The `exr` crate's own rendering of the failure.
        message: String,
    },
    /// The file carries deep data, which is one sample *list* per pixel
    /// rather than one sample.
    ///
    /// vips cannot read deep EXR either; it fails in `ImfOpenInputFile`
    /// with the OpenEXR library's own message.
    #[error("exr: deep data is not supported; the first part declares deep samples")]
    DeepData,
    /// A selected channel stores 32-bit unsigned integers.
    ///
    /// libviprs has no unsigned-integer sample carrier yet (issue #517)
    /// and `f32` cannot hold a `u32` above 2^24 exactly, so this refuses
    /// rather than silently rounding. vips converts UINT to `half` and so
    /// saturates every value above 65504 to infinity.
    #[error(
        "exr: channel {channel:?} stores 32-bit unsigned integers, which need the \
         uint sample carrier (issue #517); only HALF and FLOAT channels load"
    )]
    UnsupportedSampleType {
        /// The offending channel's name as it appears in the file.
        channel: String,
    },
    /// A selected channel is subsampled, so it carries fewer samples than
    /// the layer has pixels.
    ///
    /// Chroma subsampling is legal in flat scanline EXR files and would
    /// need a per-channel upsample to land in a single interleaved
    /// raster. Nothing in libviprs consumes it, so it is refused rather
    /// than guessed at.
    #[error(
        "exr: channel {channel:?} is subsampled {x_sampling}x{y_sampling}; only \
         1x1 channels load"
    )]
    SubsampledChannel {
        /// The offending channel's name as it appears in the file.
        channel: String,
        /// The channel's horizontal sampling rate.
        x_sampling: usize,
        /// The channel's vertical sampling rate.
        y_sampling: usize,
    },
    /// The first part declares no channels at all, so there is nothing to
    /// decode.
    #[error("exr: the first part declares no channels")]
    NoChannels,
    /// The declared data window is zero, or wider or taller than a `u32`
    /// can address.
    #[error("exr: declared data window {width}x{height} is out of bounds")]
    DimensionOutOfBounds {
        /// The declared data-window width.
        width: u64,
        /// The declared data-window height.
        height: u64,
    },
    /// The file has more selected channels than the [`PixelFormat`]
    /// carrier can name.
    ///
    /// `FloatF32(n)` holds `n` in a [`NonZeroU16`], so a layer with more
    /// than 65535 channels has no spelling here. Such a file is legal
    /// OpenEXR and vips would read four of its channels and ignore the
    /// rest.
    #[error("exr: {channels} channels exceed the {max}-band carrier ceiling")]
    TooManyChannels {
        /// How many channels the selection produced.
        channels: usize,
        /// The ceiling, `u16::MAX`.
        max: usize,
    },
    /// The layer the decoder returned is not the part the header pass
    /// priced against the decode budget.
    ///
    /// The two passes over the same bytes are asked for slightly
    /// different things: the header pass takes part zero, the one vips
    /// reads, while the decoder's `first_valid_layer` takes the first part
    /// it can decode. On a multi-part file whose first part is malformed
    /// those are not the same part, and the geometry and channel
    /// selection would then belong to a different image than the pixels.
    /// Refusing is the only safe answer, because the budget was priced
    /// against the part that was skipped.
    #[error(
        "exr: the decoder returned a {got_width}x{got_height} layer where the \
         first part declares {want_width}x{want_height}; a multi-part file whose \
         first part cannot be decoded is not supported"
    )]
    PartMismatch {
        /// The decoded layer's width.
        got_width: usize,
        /// The decoded layer's height.
        got_height: usize,
        /// The first part's declared width.
        want_width: u32,
        /// The first part's declared height.
        want_height: u32,
    },
    /// The decoder returned a channel whose sample count does not match
    /// the declared data window.
    ///
    /// Defensive: the `exr` crate validates this itself, so reaching it
    /// means the two disagree and the raster would be built from the
    /// wrong number of samples.
    #[error(
        "exr: channel {channel:?} carries {samples} samples for a {expected}-pixel \
         data window"
    )]
    ChannelSizeMismatch {
        /// The offending channel's name as it appears in the file.
        channel: String,
        /// How many samples the channel actually carries.
        samples: usize,
        /// How many the data window calls for.
        expected: usize,
    },
    /// Constructing the decoded [`Raster`] failed.
    #[error(transparent)]
    Raster(#[from] RasterError),
}

impl From<exr::error::Error> for ExrError {
    fn from(err: exr::error::Error) -> Self {
        Self::Decode {
            message: err.to_string(),
        }
    }
}

/// Which channels [`decode_exr`] picked, and what that means for the
/// interpretation tag.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Selection {
    /// Channel names in output band order.
    names: Vec<String>,
    /// The tag to put on the raster.
    interpretation: Interpretation,
}

/// Decode OpenEXR bytes into a float [`Raster`] (libvips `openexrload`).
///
/// The result is a float raster with one band per selected channel,
/// holding the file's own scene-linear samples. The format is the canonical
/// spelling of that band count, so an RGBA selection comes back as
/// [`PixelFormat::RgbaF32`] and every other count as
/// [`PixelFormat::FloatF32`]`(n)` (issue #531). HALF widens to `f32`
/// exactly and FLOAT is copied through; neither is quantised, which is
/// where this parts company with vips (see the [module docs](self)).
///
/// # Channel selection
///
/// EXR stores an arbitrary set of *named* channels, so a loader has to
/// choose. The rule, applied to the first part's channel list:
///
/// 1. All three of `R`, `G` and `B` present: bands are `R, G, B`, plus
///    `A` when the file has it. Tagged [`Interpretation::ScRgb`], the tag
///    vips uses (`openexr2vips.c:222-224`).
/// 2. Otherwise `Y` present: bands are `Y`, plus `A` when the file has
///    it. Tagged [`Interpretation::Multiband`].
/// 3. Otherwise: every channel the file declares, in the alphabetical
///    order OpenEXR stores them in. Tagged [`Interpretation::Multiband`].
///
/// Rule 3 is the one vips has no equivalent for. Measured, a file whose
/// only channel is `Z` loads in vips as four bands of `(0, 0, 0, 1)`, a
/// black image with no error, because the RGBA wrapper found nothing it
/// recognised and used its fill values. Here it loads as one band of
/// depth. The selected names are attached as `exr-channels` so a caller
/// never has to guess what a band means.
///
/// # Metadata
///
/// | field | value |
/// |---|---|
/// | `exr-channels` | the selected names, comma separated, in band order. OpenEXR does not forbid a comma inside a channel name, so treat this as a label rather than as something to split on when the names are not the usual `R`/`G`/`B`/`A` |
/// | `exr-compression` | the first part's compression method |
/// | `exr-data-window-left` / `-top` | the data-window origin, which the pixels have been normalised away from |
/// | `exr-parts` | how many parts the file has; only the first is decoded |
/// | `tile-width` / `tile-height` | set only for a tiled file, as vips does |
///
/// The part count is `exr-parts` and not the shared `n-pages` on purpose.
/// vips attaches no `n-pages` to an EXR at all, so a raster from here
/// reports [`Raster::get_n_pages`]` == 1` exactly as one loaded through
/// `openexrload` does. Beyond parity, the two counts mean different
/// things: `n-pages` is paired with a page index a caller can ask a
/// loader for ([`crate::decode_tiff_page`] takes one), while an EXR part
/// is a layer, and `decode_exr` takes no part argument, so a sweep over
/// `0..get_n_pages()` would be a sweep over something unreachable.
/// `exr-parts` says what it is and leaves the page model alone.
///
/// # Limitations
///
/// * **UINT channels do not load.** They need the unsigned sample carrier
///   from issue #517. [`ExrError::UnsupportedSampleType`] names it.
/// * **Multi-part files decode their first part only**, which is also all
///   vips can reach. `exr-parts` reports the real count.
/// * **Deep EXR does not load** ([`ExrError::DeepData`]). Neither does it
///   in vips.
/// * **Chroma-subsampled channels do not load**
///   ([`ExrError::SubsampledChannel`]).
/// * **A float raster is rejected by the pyramid engine.**
///   [`crate::resize::downscale_half`] and `downscale_to` both return
///   [`RasterError::FloatUnsupported`] for a float format, so a loaded
///   `.exr` cannot be fed to [`crate::EngineBuilder`]. The resampling
///   surface in [`crate::resample`] does handle float, so ordinary
///   operations work; only the tiled-pyramid path is closed. Cast to an
///   integer format first if you need a pyramid, and accept that doing so
///   throws the high dynamic range away.
///
/// # Errors
///
/// * [`SourceError::Exr`] wrapping any [`ExrError`] variant: a bad magic,
///   a malformation the `exr` crate rejected, deep data, a UINT or
///   subsampled channel, an empty or over-wide channel list, out-of-bounds
///   geometry, or an over-budget allocation.
/// * [`SourceError::CoordLimitExceeded`] when either data-window axis
///   exceeds [`DecodeLimits::max_coord`].
/// * [`SourceError::DimensionLimitExceeded`] when `width * height` exceeds
///   [`DecodeLimits::max_pixels`].
pub fn decode_exr(bytes: &[u8], limits: DecodeLimits) -> Result<Raster, SourceError> {
    if !bytes.starts_with(&MAGIC) {
        return Err(ExrError::BadMagic {
            found: bytes[..bytes.len().min(MAGIC.len())].to_vec(),
        }
        .into());
    }

    // Read the headers on their own first, so the geometry is checked
    // against the decode budget before a single pixel is allocated. The
    // `exr` builder in the full read below would otherwise size its
    // buffers from the same untrusted numbers.
    let meta = exr::meta::MetaData::read_from_buffered(Cursor::new(bytes), false)
        .map_err(ExrError::from)?;
    let part_count = meta.headers.len();
    let header = meta.headers.first().ok_or(ExrError::Decode {
        message: "the file declares no parts".to_owned(),
    })?;

    if header.deep {
        return Err(ExrError::DeepData.into());
    }

    let (width, height) = check_geometry(header.layer_size, limits)?;
    let selection = select_channels(&header.channels)?;
    let bands = selection.names.len();
    // `select_channels` rejects an empty list, so the only way `bands` can
    // fail to be a `NonZeroU16` is by being too large for one.
    let band_count =
        u16::try_from(bands)
            .ok()
            .and_then(NonZeroU16::new)
            .ok_or(ExrError::TooManyChannels {
                channels: bands,
                max: usize::from(u16::MAX),
            })?;

    // The price is per DECLARED channel, not per selected band. The read
    // below asks for `all_channels()`, and `exr` builds one
    // full-resolution sample buffer per channel the header declares
    // (`image/read/samples.rs`, `create_samples_level_reader`, a
    // `vec![_; resolution.area()]` for each) before it decompresses a
    // single block. So a 4096x4096 file declaring 64 FLOAT channels
    // allocates 4 GiB no matter that the selection keeps four of them, and
    // pricing off `bands` would under-count that by `declared / selected`,
    // a ratio nothing bounds. `exr` carries no `try_reserve` call at all,
    // so the top of that range is a process abort rather than an error,
    // which is exactly the failure class this budget exists to prevent.
    //
    // `SAMPLE_BYTES` is the f32 carrier's 4, and a declared HALF channel
    // only costs `exr` 2, so this over-prices half files. That is the safe
    // direction, and it is also the true price of the interleaved output
    // buffer built after the decode, which costs the selected bands again
    // on top of what `exr` is holding. The budget is therefore a floor on
    // the peak rather than the peak itself.
    //
    // The price and the comparison are the crate's, not this module's
    // (issue #632): `decode_alloc_bytes` saturates rather than wrapping,
    // which matters because `max_coord`, `max_pixels` and
    // `max_alloc_bytes` are all caller-settable and a caller who lifts
    // every ceiling must still get a refusal here rather than a wrapped
    // price that waves a huge allocation through. The price, the comparison
    // and now the reporting are all the crate's: this used to build an
    // `ExrError::AllocLimitExceeded` of its own, one of five variants
    // re-tagging the same refusal, which #686 collapsed onto
    // `SourceError::AllocLimitExceeded`.
    //
    // The band count reported is the one the header **declares**, not the
    // number of bands the selection keeps, because that is what the price is
    // and what the decoder allocates for.
    let declared = header.channels.list.len();
    //
    // `declared as u64` is lossless on every target this builds for, and it is
    // the count that prices the frame. The `u32` narrowing that used to be
    // here saturated the price *down*, which is the one direction that can
    // turn a refusal into a decode; it now happens inside `check_image_alloc`
    // and only to the geometry the message reports.
    limits.check_image_alloc(
        "OpenEXR sample buffers",
        width,
        height,
        declared as u64,
        SAMPLE_BYTES as u64,
    )?;

    // `non_parallel` is not an optimisation choice, it is the contract:
    // `exr`'s `rayon` feature is off in `Cargo.toml` because libviprs owns
    // its own scheduling, and asking for parallel decompression without it
    // would only add a code path that behaves differently under the two
    // feature sets.
    let image = exr::prelude::read()
        .no_deep_data()
        .largest_resolution_level()
        .all_channels()
        .first_valid_layer()
        .all_attributes()
        .non_parallel()
        .from_buffered(Cursor::new(bytes))
        .map_err(ExrError::from)?;
    let layer = &image.layer_data;

    // The two passes are asked for slightly different things: the header
    // pass took part zero, the part vips reads, while `first_valid_layer`
    // takes the first part it can decode. Those diverge on a multi-part
    // file whose first part is malformed, and the budget above was priced
    // against the part that would then have been skipped.
    let exr::math::Vec2(got_w, got_h) = layer.size;
    if got_w != width as usize || got_h != height as usize {
        return Err(ExrError::PartMismatch {
            got_width: got_w,
            got_height: got_h,
            want_width: width,
            want_height: height,
        }
        .into());
    }

    // Both of these were plain `usize` products. They go through
    // `buffer_len` for the same reason the price goes through
    // `decode_alloc_bytes`: clearing the budget says the byte count fits a
    // `u64`, which on a 32-bit target is not the same as fitting the
    // address space, and a caller can raise `max_alloc_bytes` past 4 GiB
    // there. `bands` is at most `u16::MAX` by the `band_count` conversion
    // above, so `bands * SAMPLE_BYTES` is the one multiply here that
    // cannot overflow on any target.
    let pixels = buffer_len(width, height, 1).map_err(ExrError::Raster)?;
    let mut data =
        vec![0u8; buffer_len(width, height, bands * SAMPLE_BYTES).map_err(ExrError::Raster)?];
    for (band, name) in selection.names.iter().enumerate() {
        let channel = layer
            .channel_data
            .list
            .iter()
            .find(|c| c.name.as_slice() == name.as_bytes())
            .ok_or(ExrError::PartMismatch {
                got_width: got_w,
                got_height: got_h,
                want_width: width,
                want_height: height,
            })?;
        write_channel(&mut data, &channel.sample_data, name, band, bands, pixels)?;
    }

    // The canonical spelling of the layout: a four-channel selection is
    // `RgbaF32`, not `FloatF32(4)`, which is the same pixel layout under a
    // second name that disagrees with the first about `has_alpha` (issue
    // #531). `Raster::new` would canonicalise it anyway; saying so here
    // keeps the line honest about what it produces.
    let format = PixelFormat::with_channels(usize::from(band_count.get()), SAMPLE_BYTES)
        .expect("a non-zero band count at the 4-byte float depth is a valid format");
    let mut raster = Raster::new(width, height, format, data).map_err(ExrError::Raster)?;
    raster.meta.interpretation = Some(selection.interpretation);
    attach_fields(&mut raster, header, &selection, part_count);
    Ok(raster)
}

/// Validate the declared data window against the decode budget.
///
/// Returns the window as a `u32` pair so the rest of the decode never has
/// to re-narrow it.
fn check_geometry(
    size: exr::math::Vec2<usize>,
    limits: DecodeLimits,
) -> Result<(u32, u32), SourceError> {
    let exr::math::Vec2(w, h) = size;
    let (Ok(width), Ok(height)) = (u32::try_from(w), u32::try_from(h)) else {
        return Err(ExrError::DimensionOutOfBounds {
            width: w as u64,
            height: h as u64,
        }
        .into());
    };
    if width == 0 || height == 0 {
        return Err(ExrError::DimensionOutOfBounds {
            width: u64::from(width),
            height: u64::from(height),
        }
        .into());
    }
    limits.check_coord(width, height)?;
    limits.check_pixels(width, height)?;
    Ok((width, height))
}

/// Apply the three-rule channel selection described on [`decode_exr`].
fn select_channels(channels: &exr::meta::attribute::ChannelList) -> Result<Selection, ExrError> {
    let names: Vec<String> = channels.list.iter().map(|c| c.name.to_string()).collect();
    if names.is_empty() {
        return Err(ExrError::NoChannels);
    }
    let has = |want: &str| names.iter().any(|n| n == want);

    let (selected, interpretation) = if has(CHANNEL_R) && has(CHANNEL_G) && has(CHANNEL_B) {
        let mut v = vec![
            CHANNEL_R.to_owned(),
            CHANNEL_G.to_owned(),
            CHANNEL_B.to_owned(),
        ];
        if has(CHANNEL_A) {
            v.push(CHANNEL_A.to_owned());
        }
        // The tag vips puts on every EXR it loads, and the one
        // `Raster::colourspace` reads to know these samples are
        // linear-light rather than gamma-encoded.
        (v, Interpretation::ScRgb)
    } else if has(CHANNEL_Y) {
        let mut v = vec![CHANNEL_Y.to_owned()];
        if has(CHANNEL_A) {
            v.push(CHANNEL_A.to_owned());
        }
        (v, Interpretation::Multiband)
    } else {
        // No recognised colour channels. vips would hand back a black
        // four-band image here; carrying the file's own channels instead
        // is the whole reason this module does not go through the RGBA
        // wrapper.
        (names.clone(), Interpretation::Multiband)
    };

    // Reject before decoding, not after, so an unsupported file costs a
    // header parse rather than a full decompression pass.
    for name in &selected {
        let Some(desc) = channels
            .list
            .iter()
            .find(|c| c.name.as_slice() == name.as_bytes())
        else {
            continue;
        };
        if desc.sample_type == exr::meta::attribute::SampleType::U32 {
            return Err(ExrError::UnsupportedSampleType {
                channel: name.clone(),
            });
        }
        let exr::math::Vec2(sx, sy) = desc.sampling;
        if sx != 1 || sy != 1 {
            return Err(ExrError::SubsampledChannel {
                channel: name.clone(),
                x_sampling: sx,
                y_sampling: sy,
            });
        }
    }

    Ok(Selection {
        names: selected,
        interpretation,
    })
}

/// Widen one decoded channel into its band of the interleaved output.
///
/// `F16` widens exactly, `F32` is copied through, and `U32` is
/// unreachable because [`select_channels`] rejects it before the decode
/// runs.
fn write_channel(
    data: &mut [u8],
    samples: &exr::image::FlatSamples,
    name: &str,
    band: usize,
    bands: usize,
    pixels: usize,
) -> Result<(), ExrError> {
    let len = samples.len();
    if len != pixels {
        return Err(ExrError::ChannelSizeMismatch {
            channel: name.to_owned(),
            samples: len,
            expected: pixels,
        });
    }
    let mut put = |i: usize, v: f32| {
        let off = (i * bands + band) * SAMPLE_BYTES;
        data[off..off + SAMPLE_BYTES].copy_from_slice(&v.to_ne_bytes());
    };
    match samples {
        exr::image::FlatSamples::F16(v) => {
            for (i, s) in v.iter().enumerate() {
                put(i, s.to_f32());
            }
        }
        exr::image::FlatSamples::F32(v) => {
            for (i, s) in v.iter().enumerate() {
                put(i, *s);
            }
        }
        exr::image::FlatSamples::U32(_) => {
            return Err(ExrError::UnsupportedSampleType {
                channel: name.to_owned(),
            });
        }
    }
    Ok(())
}

/// Attach the header facts that survive the decode.
fn attach_fields(
    raster: &mut Raster,
    header: &exr::meta::header::Header,
    selection: &Selection,
    part_count: usize,
) {
    raster.fields.set(
        "exr-channels",
        MetadataValue::Str(selection.names.join(",")),
    );
    raster.fields.set(
        "exr-compression",
        MetadataValue::Str(compression_name(header.compression).to_owned()),
    );
    let exr::math::Vec2(left, top) = header.own_attributes.layer_position;
    raster
        .fields
        .set("exr-data-window-left", MetadataValue::Int(i64::from(left)));
    raster
        .fields
        .set("exr-data-window-top", MetadataValue::Int(i64::from(top)));
    // `exr-parts`, not `n-pages`. vips attaches no page count to an EXR,
    // and an EXR part is a layer rather than a page: `decode_exr` has no
    // part selector, so a count read back through `Raster::get_n_pages`
    // would promise an iteration this loader cannot serve.
    raster
        .fields
        .set("exr-parts", MetadataValue::Int(part_count as i64));
    // vips sets these two for a tiled file and leaves them absent for a
    // scanline one (`read_header`, `openexr2vips.c:229-232`), so the
    // presence of the field is itself the signal.
    if let exr::meta::BlockDescription::Tiles(tiles) = header.blocks {
        let exr::math::Vec2(tw, th) = tiles.tile_size;
        raster
            .fields
            .set("tile-width", MetadataValue::Int(tw as i64));
        raster
            .fields
            .set("tile-height", MetadataValue::Int(th as i64));
    }
}

/// The spelling `exrheader` uses for a compression method, so a captured
/// oracle value and a libviprs field read the same.
const fn compression_name(compression: exr::compression::Compression) -> &'static str {
    use exr::compression::Compression;
    match compression {
        Compression::Uncompressed => "none",
        Compression::RLE => "rle",
        Compression::ZIP1 => "zips",
        Compression::ZIP16 => "zip",
        Compression::PIZ => "piz",
        Compression::PXR24 => "pxr24",
        Compression::B44 => "b44",
        Compression::B44A => "b44a",
        Compression::DWAA(_) => "dwaa",
        Compression::DWAB(_) => "dwab",
        // High-Throughput JPEG 2000, added to the format in OpenEXR 3.4.
        // `exr` 1.74.2 names the two block sizes but decodes neither, so a
        // file using them fails in the read below with the crate's own
        // "not supported" message; this arm only exists so the field is
        // still spelled if that changes.
        Compression::HTJ2K32 => "ht-j2k-32",
        Compression::HTJ2K256 => "ht-j2k-256",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::DeclaredGeometry;

    /// Every fixture is written by the OpenEXR reference implementation
    /// 3.4.15 and lives beside the capture script that measured vips on
    /// it. See `oracle-captures/foreign-exr/make_corpus.cpp`.
    fn fixture(name: &str) -> Vec<u8> {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/oracle-captures/foreign-exr/fixtures/"
        );
        std::fs::read(format!("{path}{name}.exr"))
            .unwrap_or_else(|e| panic!("fixture {name}.exr must be readable: {e}"))
    }

    /// Read a decoded raster back as `f32` samples in band-interleaved
    /// order, which is the shape every oracle value in this file is in.
    fn samples(raster: &Raster) -> Vec<f32> {
        let data = raster.data();
        (0..data.len() / SAMPLE_BYTES)
            .map(|i| {
                let mut raw = [0u8; SAMPLE_BYTES];
                raw.copy_from_slice(&data[i * SAMPLE_BYTES..(i + 1) * SAMPLE_BYTES]);
                f32::from_ne_bytes(raw)
            })
            .collect()
    }

    /// The corpus ramp: band `b` of pixel `(x, y)` in an 8-wide image is
    /// `(x + y * 8 + b * 7) * step`. This is the generator's `ramp()`,
    /// restated so a failure names the value it expected rather than a
    /// blob.
    fn ramp(x: usize, y: usize, width: usize, b: usize, step: f32) -> f32 {
        (x + y * width + b * 7) as f32 * step
    }

    /**
     * Tests that a ZIP-compressed RGBA half file decodes to the exact
     * samples the OpenEXR reference writer put in it, with no tolerance.
     * Works by rebuilding the generator's ramp and comparing every
     * sample with `assert_eq!` on the bit pattern, which is legitimate
     * here only because half widens to f32 exactly.
     * Input: `rgba_half_zip.exr`, 8x4 RGBA half -> Output: `RgbaF32`,
     * 128 samples, `(0.0, 3.5, 7.0, 10.5)` at pixel (0, 0).
     */
    #[test]
    fn rgba_half_decodes_exactly() {
        let raster = decode_exr(&fixture("rgba_half_zip"), DecodeLimits::default()).unwrap();
        assert_eq!((raster.width(), raster.height()), (8, 4));
        assert_eq!(raster.format().channels(), 4);
        assert!(raster.format().is_float());
        assert_eq!(raster.interpretation(), Interpretation::ScRgb);

        let got = samples(&raster);
        assert_eq!(got.len(), 8 * 4 * 4);
        for y in 0..4 {
            for x in 0..8 {
                for b in 0..4 {
                    let want = ramp(x, y, 8, b, 0.5);
                    let idx = (y * 8 + x) * 4 + b;
                    assert_eq!(
                        got[idx], want,
                        "sample ({x}, {y}) band {b}: half widens to f32 exactly, \
                         so this must be equal and not merely close"
                    );
                }
            }
        }
    }

    /**
     * Tests that the six lossless compression methods all decode to the
     * identical payload, which is what makes the vips parity pins in
     * this module exact rather than toleranced.
     * Works by decoding each fixture and comparing its whole byte buffer
     * with the ZIP one. NONE, RLE, ZIPS, ZIP and PIZ are lossless for
     * every sample type; PXR24 is lossless for HALF specifically,
     * because its 24-bit truncation only bites `f32`.
     * Input: the `rgba_half_*` compression sweep -> Output: six
     * byte-identical rasters.
     */
    #[test]
    fn lossless_compressions_agree_byte_for_byte() {
        let reference = decode_exr(&fixture("rgba_half_zip"), DecodeLimits::default()).unwrap();
        for name in [
            "rgba_half_none",
            "rgba_half_rle",
            "rgba_half_zips",
            "rgba_half_piz",
            "rgba_half_pxr24",
        ] {
            let raster = decode_exr(&fixture(name), DecodeLimits::default()).unwrap();
            assert_eq!(
                raster.data(),
                reference.data(),
                "{name} is a lossless coding of the same payload, so it must \
                 decode byte for byte to the ZIP result"
            );
        }
    }

    /**
     * Tests that a tiled file decodes to the same pixels as the
     * equivalent scanline file, and carries the tile geometry vips
     * attaches.
     * Works by comparing against the scanline ZIP fixture and reading
     * back the two metadata fields.
     * Input: `rgba_half_tiled.exr`, 8x4 in 4x2 tiles -> Output: the ZIP
     * payload plus `tile-width` 4 and `tile-height` 2.
     */
    #[test]
    fn tiled_matches_scanline_and_reports_tile_size() {
        let reference = decode_exr(&fixture("rgba_half_zip"), DecodeLimits::default()).unwrap();
        let tiled = decode_exr(&fixture("rgba_half_tiled"), DecodeLimits::default()).unwrap();
        assert_eq!(tiled.data(), reference.data());
        assert_eq!(tiled.get_field("tile-width"), Some(MetadataValue::Int(4)));
        assert_eq!(tiled.get_field("tile-height"), Some(MetadataValue::Int(2)));
        assert_eq!(reference.get_field("tile-width"), None);
    }

    /**
     * Tests that a tiled file whose image size is not a whole number of
     * tiles decodes its ragged edge correctly.
     * Works by rebuilding the ramp for a 7x5 image, which 4x4 tiles
     * cover with a 1-pixel right column and a 1-pixel bottom row.
     * Input: `rgba_half_tiled_ragged.exr` -> Output: 7x5 `RgbaF32`
     * matching the ramp at every pixel including the partial tiles.
     */
    #[test]
    fn ragged_tiles_decode_their_edges() {
        let raster =
            decode_exr(&fixture("rgba_half_tiled_ragged"), DecodeLimits::default()).unwrap();
        assert_eq!((raster.width(), raster.height()), (7, 5));
        let got = samples(&raster);
        for y in 0..5 {
            for x in 0..7 {
                for b in 0..4 {
                    let want = ramp(x, y, 7, b, 0.5);
                    assert_eq!(got[(y * 7 + x) * 4 + b], want, "({x}, {y}) band {b}");
                }
            }
        }
    }

    /**
     * Tests the divergence this module exists for: a FLOAT channel keeps
     * its `f32` value here, where vips rounds it to half.
     * Works by decoding a file whose samples are thirds, which have no
     * exact half spelling, and asserting the libviprs value is the true
     * `f32` while the vips oracle value is the half-rounded one.
     * Input: `rgba_float_fine.exr`, samples `(x + y*8 + b*7) / 3` ->
     * Output: `2.3333333` at band 1 of pixel (0, 0), where the captured
     * `vips rawsave` byte gives `2.333984375`.
     */
    #[test]
    fn float_channels_keep_full_precision_where_vips_rounds_to_half() {
        let raster = decode_exr(&fixture("rgba_float_fine"), DecodeLimits::default()).unwrap();
        let got = samples(&raster);

        // The generator computes `(x + y*width + b*7) as float * step` in
        // `f32` with `step = 1.0f / 3.0f`, so the stored sample is the
        // product of two `f32`s and not the `f32` nearest to 7/3. Spelling
        // it the generator's way is what makes this an exactness check
        // rather than a near-miss.
        let want = 7.0f32 * (1.0f32 / 3.0f32);
        assert_eq!(
            got[1], want,
            "a FLOAT channel is copied through, not quantised"
        );

        // Captured from `vips rawsave fixtures/rgba_float_fine.exr` on
        // 8.18.4: the RGBA wrapper stores the sample as half and widens
        // it again, so the file's 7/3 comes back as f16::from_f32(7/3).
        let vips_value = 2.333_984_4f32;
        assert_ne!(
            got[1], vips_value,
            "this test is the whole point of not going through ImfRgba; if it \
             passes trivially the fixture has stopped discriminating"
        );
        let half_of_want = f32::from(exr::prelude::f16::from_f32(want));
        assert_eq!(
            half_of_want, vips_value,
            "the vips value must be exactly the half rounding of the true sample, \
             which is what makes the divergence attributable rather than noise"
        );
    }

    /**
     * Tests that a file with R, G and B but no alpha decodes to three
     * bands rather than four.
     * Works by decoding the RGB fixture and checking the band count and
     * the channel list; vips would report four bands with a synthesised
     * alpha of 1.0, which is the divergence recorded here.
     * Input: `rgb_half_zip.exr` -> Output: `FloatF32(3)`, `exr-channels`
     * `"R,G,B"`.
     */
    #[test]
    fn rgb_without_alpha_stays_three_bands() {
        let raster = decode_exr(&fixture("rgb_half_zip"), DecodeLimits::default()).unwrap();
        assert_eq!(raster.format().channels(), 3);
        assert_eq!(
            raster.get_field("exr-channels"),
            Some(MetadataValue::Str("R,G,B".to_owned()))
        );
        let got = samples(&raster);
        assert_eq!(&got[..3], &[0.0, 3.5, 7.0]);
    }

    /**
     * Tests that a luminance-only file decodes to one band.
     * Works by decoding the `Y` fixture; vips replicates `Y` across
     * three bands and adds an alpha of 1.0, so this is a deliberate
     * divergence and the assertion names it.
     * Input: `y_half_zip.exr` -> Output: `FloatF32(1)` tagged
     * `Multiband`, first sample `0.0`, second `0.5`.
     */
    #[test]
    fn luminance_only_file_stays_one_band() {
        let raster = decode_exr(&fixture("y_half_zip"), DecodeLimits::default()).unwrap();
        assert_eq!(raster.format().channels(), 1);
        assert_eq!(raster.interpretation(), Interpretation::Multiband);
        assert_eq!(
            raster.get_field("exr-channels"),
            Some(MetadataValue::Str("Y".to_owned()))
        );
        let got = samples(&raster);
        assert_eq!(&got[..2], &[0.0, 0.5]);
    }

    /**
     * Tests that a file with no colour channels at all still carries its
     * data, which is the case vips loses outright.
     * Works by decoding a single-channel depth pass. Measured on vips
     * 8.18.4, the same file loads as four bands of `(0, 0, 0, 1)`: the
     * RGBA wrapper recognises nothing and returns its fill values, with
     * no error and no warning.
     * Input: `z_float_zip.exr`, one FLOAT channel named `Z` -> Output:
     * `FloatF32(1)` holding the ramp, `exr-channels` `"Z"`.
     */
    #[test]
    fn depth_only_file_survives_where_vips_returns_black() {
        let raster = decode_exr(&fixture("z_float_zip"), DecodeLimits::default()).unwrap();
        assert_eq!(raster.format().channels(), 1);
        assert_eq!(
            raster.get_field("exr-channels"),
            Some(MetadataValue::Str("Z".to_owned()))
        );
        let got = samples(&raster);
        assert_eq!(&got[..3], &[0.0, 0.5, 1.0]);
        assert!(
            got.iter().any(|s| *s != 0.0),
            "vips decodes this file to all zeros; the point of the name-based \
             selection is that libviprs does not"
        );
    }

    /**
     * Tests that a data window away from the origin decodes to the same
     * pixels at (0, 0), and that the origin it dropped is recorded.
     * Works by comparing against the origin-anchored fixture, which the
     * generator wrote from the identical payload. vips normalises the
     * same way (`openexr2vips.c:401-403`) but keeps no record of the
     * offset.
     * Input: `rgba_half_offset.exr`, data window at (5, 7) -> Output:
     * the ZIP payload with `exr-data-window-left` 5 and `-top` 7.
     */
    #[test]
    fn offset_data_window_normalises_to_the_origin() {
        let reference = decode_exr(&fixture("rgba_half_zip"), DecodeLimits::default()).unwrap();
        let offset = decode_exr(&fixture("rgba_half_offset"), DecodeLimits::default()).unwrap();
        assert_eq!((offset.width(), offset.height()), (8, 4));
        assert_eq!(offset.data(), reference.data());
        assert_eq!(
            offset.get_field("exr-data-window-left"),
            Some(MetadataValue::Int(5))
        );
        assert_eq!(
            offset.get_field("exr-data-window-top"),
            Some(MetadataValue::Int(7))
        );
    }

    /**
     * Tests that the display window does not size the image; the data
     * window does.
     * Works by decoding a file whose display window is 16x16 and whose
     * data window is 8x4 at (2, 3). vips sizes from the data window too
     * (`read_new`, `openexr2vips.c:186-191`), measured as 8x4.
     * Input: `rgba_half_display.exr` -> Output: an 8x4 raster.
     */
    #[test]
    fn display_window_does_not_size_the_image() {
        let raster = decode_exr(&fixture("rgba_half_display"), DecodeLimits::default()).unwrap();
        assert_eq!((raster.width(), raster.height()), (8, 4));
    }

    /**
     * Tests that a UINT channel is refused with a typed error naming the
     * carrier issue rather than being silently widened.
     * Works by matching the structured variant and its channel field.
     * vips loads the same file by converting UINT to half, which
     * saturates every value above 65504 to infinity.
     * Input: `rgba_uint_zip.exr` -> Output:
     * `ExrError::UnsupportedSampleType { channel: "A" }`, `A` being first
     * in the alphabetical channel list the selection walks.
     */
    #[test]
    fn uint_channels_are_refused_by_variant() {
        let err = decode_exr(&fixture("rgba_uint_zip"), DecodeLimits::default()).unwrap_err();
        assert!(
            matches!(
                &err,
                SourceError::Exr(ExrError::UnsupportedSampleType { channel })
                    if channel == CHANNEL_R
            ),
            "expected UnsupportedSampleType for the first selected channel, got {err:?}"
        );
        assert!(
            err.to_string().contains("#517"),
            "the message must name the tracking issue so the ceiling is \
             actionable, got {err}"
        );
    }

    /**
     * Tests that the alloc budget is checked against the declared data
     * window before any pixel buffer is allocated.
     * Works by decoding a legitimate 8x4x4 file under a budget one byte
     * short of the 512 bytes it needs, and matching the structured
     * variant's arithmetic.
     * Input: `rgba_half_zip.exr` with `max_alloc_bytes = 511` -> Output:
     * `AllocLimitExceeded { needed: 512, .. }`.
     */
    #[test]
    fn alloc_budget_is_checked_before_decoding() {
        let limits = DecodeLimits::default().with_max_alloc_bytes(511);
        let err = decode_exr(&fixture("rgba_half_zip"), limits).unwrap_err();
        assert!(
            matches!(
                err,
                SourceError::AllocLimitExceeded {
                    what: "OpenEXR sample buffers",
                    geometry: Some(DeclaredGeometry {
                        width: 8,
                        height: 4,
                        bands: 4,
                    }),
                    needed_bytes: 512,
                    max_alloc_bytes: 511,
                }
            ),
            "expected the 8x4x4 geometry to be priced at 512 bytes, got {err:?}"
        );
    }

    /**
     * Tests that the budget bites at exactly the byte the declared window
     * costs, and not one byte either side. The case above pins only the
     * refusing half, at 511 against a price of 512; a price wrong by a
     * factor would refuse there too. This one adds the half that a wrong
     * price cannot survive.
     * Input: `rgba_half_zip.exr` at `max_alloc_bytes` 512 then 511 ->
     * Output: a clean 8x4 four-band decode, then `AllocLimitExceeded`.
     */
    #[test]
    fn the_window_budget_bites_at_exactly_the_declared_price() {
        let exact = DecodeLimits::default().with_max_alloc_bytes(512);
        let raster = decode_exr(&fixture("rgba_half_zip"), exact)
            .expect("512 bytes is exactly the 8x4x4 declared window");
        assert_eq!((raster.width(), raster.height()), (8, 4));
        assert_eq!(raster.format().channels(), 4);

        let short = DecodeLimits::default().with_max_alloc_bytes(511);
        assert!(matches!(
            decode_exr(&fixture("rgba_half_zip"), short),
            Err(SourceError::AllocLimitExceeded {
                needed_bytes: 512,
                ..
            })
        ));
    }

    /**
     * Tests that the alloc budget is priced off the channels the header
     * DECLARES and not off the bands the selection keeps, which is what
     * the decoder actually allocates for.
     * Works by decoding a 16-channel 8x4 file whose selection is the four
     * R/G/B/A bands, under a 1024-byte budget. Priced off the selection
     * that is 8*4*4*4 = 512 bytes and passes; priced off the sixteen
     * declared channels it is 8*4*16*4 = 2048 and must not. `exr` builds
     * one full-resolution buffer per declared channel before it
     * decompresses anything, and it has no `try_reserve` anywhere, so an
     * under-count here is an abort rather than an error on a file that
     * declares enough channels.
     * Input: `rgba_aov_half_zip.exr` (16 declared, 4 selected) with
     * `max_alloc_bytes = 1024` -> Output: `AllocLimitExceeded { channels:
     * 16, needed: 2048, .. }`, with `channels` naming the count the price
     * was computed from so the message is not self-contradicting.
     */
    #[test]
    fn alloc_budget_prices_every_declared_channel_not_only_the_selected_ones() {
        let selected_price = 8 * 4 * 4 * SAMPLE_BYTES as u64;
        assert_eq!(
            selected_price, 512,
            "the four selected bands cost this much"
        );
        let limits = DecodeLimits::default().with_max_alloc_bytes(1024);
        assert!(
            selected_price <= 1024,
            "the budget has to sit above the selected price, or the test would \
             pass for the wrong reason"
        );

        let err = decode_exr(&fixture("rgba_aov_half_zip"), limits).unwrap_err();
        assert!(
            matches!(
                err,
                SourceError::AllocLimitExceeded {
                    what: "OpenEXR sample buffers",
                    geometry: Some(DeclaredGeometry {
                        width: 8,
                        height: 4,
                        bands: 16,
                    }),
                    needed_bytes: 2048,
                    max_alloc_bytes: 1024,
                }
            ),
            "expected the sixteen declared channels to be priced at 2048 bytes, \
             got {err:?}"
        );

        // And the same file decodes to the four selected bands once the
        // budget covers the declared price, so this is a budget rule and
        // not a refusal to read a multi-AOV render.
        let raster = decode_exr(&fixture("rgba_aov_half_zip"), DecodeLimits::default()).unwrap();
        assert_eq!((raster.width(), raster.height()), (8, 4));
        assert_eq!(raster.format().channels(), 4);
        assert!(raster.format().is_float());
        assert_eq!(
            raster.get_field("exr-channels"),
            Some(MetadataValue::Str("R,G,B,A".to_owned()))
        );
    }

    /**
     * Tests that the coordinate ceiling is enforced on the declared data
     * window, the same budget every other decoder applies.
     * Works by lowering `max_coord` below the fixture's width.
     * Input: `rgba_half_zip.exr` (8x4) with `max_coord = 4` -> Output:
     * `SourceError::CoordLimitExceeded`.
     */
    #[test]
    fn coord_ceiling_is_enforced_on_the_data_window() {
        let limits = DecodeLimits::default().with_max_coord(4);
        let err = decode_exr(&fixture("rgba_half_zip"), limits).unwrap_err();
        assert!(
            matches!(err, SourceError::CoordLimitExceeded { width: 8, .. }),
            "got {err:?}"
        );
    }

    /**
     * Tests that anything without the four magic bytes is rejected
     * before the `exr` crate is even asked, so a misrouted buffer gives
     * a libviprs error rather than a decoder message.
     * Works by feeding a PNG signature and an empty slice and matching
     * the variant in both cases.
     * Input: `\x89PNG...` and `b""` -> Output: `ExrError::BadMagic` with
     * the bytes that were actually there.
     */
    #[test]
    fn bad_magic_is_rejected_with_the_bytes_it_saw() {
        let err = decode_exr(b"\x89PNG\r\n\x1a\n", DecodeLimits::default()).unwrap_err();
        assert!(
            matches!(&err, SourceError::Exr(ExrError::BadMagic { found })
                if found == &[0x89, b'P', b'N', b'G']),
            "got {err:?}"
        );
        let err = decode_exr(b"", DecodeLimits::default()).unwrap_err();
        assert!(
            matches!(&err, SourceError::Exr(ExrError::BadMagic { found }) if found.is_empty()),
            "got {err:?}"
        );
    }

    /**
     * Tests that a file truncated after its magic fails with the decode
     * variant rather than panicking or hanging.
     * Works by cutting a valid fixture to progressively shorter
     * prefixes, every one of which is untrusted input the loader has to
     * survive. This is the shape the fuzz target generalises.
     * Input: `rgba_half_zip.exr` cut to 4, 8, 16, ... bytes -> Output:
     * an error every time, never a panic.
     */
    #[test]
    fn truncated_files_error_rather_than_panic() {
        let full = fixture("rgba_half_zip");
        let mut cut = 4;
        while cut < full.len() {
            let err = decode_exr(&full[..cut], DecodeLimits::default());
            assert!(err.is_err(), "a {cut}-byte prefix must not decode");
            cut *= 2;
        }
    }

    /**
     * Tests that the smallest legal image, a single pixel, decodes.
     * Works by decoding the 1x1 fixture and reading its four samples.
     * Input: `rgba_half_1x1.exr` -> Output: `RgbaF32`, one pixel
     * `(0.0, 3.5, 7.0, 10.5)`.
     */
    #[test]
    fn single_pixel_image_decodes() {
        let raster = decode_exr(&fixture("rgba_half_1x1"), DecodeLimits::default()).unwrap();
        assert_eq!((raster.width(), raster.height()), (1, 1));
        assert_eq!(samples(&raster), vec![0.0, 3.5, 7.0, 10.5]);
    }

    /**
     * Tests that the compression method and part count reach the caller
     * as metadata, since neither survives into the pixels, and that the
     * part count travels under `exr-parts` rather than the shared
     * `n-pages`.
     * Works by reading all three fields back off a single-part ZIP file.
     * `exr-parts` is how the multi-part ceiling is made visible: vips
     * reports nothing at all and silently decodes part zero. It is not
     * `n-pages` because `vipsheader -a` attaches none to an EXR (measured,
     * `oracle-captures/foreign-exr/oracle.json`) and because an EXR part
     * is a layer, not a page a caller can ask this loader for.
     * Input: `rgba_half_zip.exr` -> Output: `exr-compression` `"zip"`,
     * `exr-parts` 1, no `n-pages` field, and `get_n_pages()` 1 as vips
     * reports for the same file.
     */
    #[test]
    fn compression_and_part_count_are_attached() {
        let raster = decode_exr(&fixture("rgba_half_zip"), DecodeLimits::default()).unwrap();
        assert_eq!(
            raster.get_field("exr-compression"),
            Some(MetadataValue::Str("zip".to_owned()))
        );
        assert_eq!(raster.get_field("exr-parts"), Some(MetadataValue::Int(1)));
        assert_eq!(
            raster.get_field("n-pages"),
            None,
            "an EXR carries no page count, matching `vipsheader -a`, which \
             reports none for any of the fixtures"
        );
        assert_eq!(
            raster.get_n_pages(),
            1,
            "the shared accessor must read 1 for an EXR, the same value it \
             reads for a raster vips loaded through openexrload"
        );
        let piz = decode_exr(&fixture("rgba_half_piz"), DecodeLimits::default()).unwrap();
        assert_eq!(
            piz.get_field("exr-compression"),
            Some(MetadataValue::Str("piz".to_owned()))
        );
    }

    /// Project a libviprs decode through the RGBA-half funnel vips reads
    /// every EXR through, so the two can be compared as whole payloads.
    ///
    /// This is `ImfRgbaFile`'s behaviour, measured rather than assumed and
    /// recorded in `oracle-captures/foreign-exr/oracle.json`: four bands
    /// out regardless of the file, `R`/`G`/`B` taken when present, `Y`
    /// replicated across all three when it is the only luminance channel,
    /// a missing alpha filled with `1.0`, an unrecognised channel set
    /// filled with `(0, 0, 0, 1)`, and every sample rounded to `half` on
    /// the way through.
    fn through_the_vips_rgba_funnel(raster: &Raster, channels: &str) -> Vec<u8> {
        let names: Vec<&str> = channels.split(',').collect();
        let got = samples(raster);
        let bands = names.len();
        let pixels = got.len() / bands;
        let band_of = |want: &str| names.iter().position(|n| *n == want);
        let (r, g, b) = match (band_of("R"), band_of("G"), band_of("B")) {
            (Some(r), Some(g), Some(b)) => (Some(r), Some(g), Some(b)),
            // The luminance case: vips hands back `(y, y, y, ...)`.
            _ => match band_of("Y") {
                Some(y) => (Some(y), Some(y), Some(y)),
                None => (None, None, None),
            },
        };
        let a = band_of("A");

        let mut out = Vec::with_capacity(pixels * 4 * SAMPLE_BYTES);
        for i in 0..pixels {
            let pick = |band: Option<usize>, fill: f32| match band {
                Some(band) => got[i * bands + band],
                None => fill,
            };
            for sample in [pick(r, 0.0), pick(g, 0.0), pick(b, 0.0), pick(a, 1.0)] {
                // The funnel's own quantisation: `ImfRgba` holds `half`.
                let rounded = f32::from(exr::prelude::f16::from_f32(sample));
                out.extend_from_slice(&rounded.to_ne_bytes());
            }
        }
        out
    }

    /**
     * Pins libviprs against the vips oracle as a whole payload, not as a
     * spot check: for every fixture the loader accepts, projecting the
     * decode through vips's RGBA-half funnel reproduces the exact bytes
     * `vips rawsave` wrote, digest for digest, with no tolerance
     * anywhere.
     * Works by decoding each fixture, applying the measured funnel
     * (four bands, R/G/B or replicated Y, alpha filled with 1.0,
     * everything rounded to half) and comparing the SHA-256 with the one
     * captured in `oracle-captures/foreign-exr/oracle.json`. That the
     * *only* thing standing between the two decoders is that funnel is
     * the whole claim of this module: the lossless codings carry the
     * samples exactly, so any residual difference would be a real bug and
     * not rounding.
     * Input: twenty reference-written fixtures -> Output: the captured
     * vips digest for each.
     */
    #[test]
    fn vips_parity_holds_exactly_once_the_rgba_funnel_is_applied() {
        use sha2::{Digest, Sha256};

        // Captured with `python3 capture.py` in
        // `oracle-captures/foreign-exr/` against vips 8.18.4, from
        // fixtures written by OpenEXR 3.4.15. `vips_payload_sha256` in
        // oracle.json is the same value.
        let cases: [(&str, &str, &str); 20] = [
            // The lossless sweep, all one payload.
            (
                "rgba_half_none",
                "R,G,B,A",
                "6aeb8bf8858b1fcbd4e99c752dd75fe3bbee77feaacb38fe5f25b31f4890afd7",
            ),
            (
                "rgba_half_rle",
                "R,G,B,A",
                "6aeb8bf8858b1fcbd4e99c752dd75fe3bbee77feaacb38fe5f25b31f4890afd7",
            ),
            (
                "rgba_half_zips",
                "R,G,B,A",
                "6aeb8bf8858b1fcbd4e99c752dd75fe3bbee77feaacb38fe5f25b31f4890afd7",
            ),
            (
                "rgba_half_zip",
                "R,G,B,A",
                "6aeb8bf8858b1fcbd4e99c752dd75fe3bbee77feaacb38fe5f25b31f4890afd7",
            ),
            (
                "rgba_half_piz",
                "R,G,B,A",
                "6aeb8bf8858b1fcbd4e99c752dd75fe3bbee77feaacb38fe5f25b31f4890afd7",
            ),
            (
                "rgba_half_pxr24",
                "R,G,B,A",
                "6aeb8bf8858b1fcbd4e99c752dd75fe3bbee77feaacb38fe5f25b31f4890afd7",
            ),
            // Lossy, and still bit-identical: the `exr` crate and the
            // OpenEXR C++ library agree on these decoders too.
            (
                "rgba_half_b44",
                "R,G,B,A",
                "ae81530d4363c851bb11ed73a2ad7b84bae23004df039a587fed5ec3d90d325b",
            ),
            (
                "rgba_half_b44a",
                "R,G,B,A",
                "ae81530d4363c851bb11ed73a2ad7b84bae23004df039a587fed5ec3d90d325b",
            ),
            (
                "rgba_half_dwaa",
                "R,G,B,A",
                "6aeb8bf8858b1fcbd4e99c752dd75fe3bbee77feaacb38fe5f25b31f4890afd7",
            ),
            (
                "rgba_half_dwab",
                "R,G,B,A",
                "6aeb8bf8858b1fcbd4e99c752dd75fe3bbee77feaacb38fe5f25b31f4890afd7",
            ),
            // Tiled, including a ragged edge.
            (
                "rgba_half_tiled",
                "R,G,B,A",
                "6aeb8bf8858b1fcbd4e99c752dd75fe3bbee77feaacb38fe5f25b31f4890afd7",
            ),
            (
                "rgba_half_tiled_ragged",
                "R,G,B,A",
                "3a38f12ef1667460c681d4480a75ac06bc0ade95d78b2d8b80612752a4c59576",
            ),
            // Geometry.
            (
                "rgba_half_offset",
                "R,G,B,A",
                "6aeb8bf8858b1fcbd4e99c752dd75fe3bbee77feaacb38fe5f25b31f4890afd7",
            ),
            (
                "rgba_half_display",
                "R,G,B,A",
                "6aeb8bf8858b1fcbd4e99c752dd75fe3bbee77feaacb38fe5f25b31f4890afd7",
            ),
            (
                "rgba_half_1x1",
                "R,G,B,A",
                "ebf864659dcb9186bc672338d297fa38d954c2a67e475e582a28a54c00ae0a5b",
            ),
            // Channel selection: the alpha fill and the Y replication.
            (
                "rgb_half_zip",
                "R,G,B",
                "86265b6a58ba9589c3d392c23487807c9b9472d639baed99f134bf06b13de2ff",
            ),
            (
                "y_half_zip",
                "Y",
                "e86796bead796bc30d8755bfcc5b8383bf771956780bf6128343e6a2109202a6",
            ),
            // Sixteen declared channels, four of them selectable. The RGBA
            // wrapper takes R/G/B/A and drops the twelve AOVs, so vips
            // lands back on the lossless payload and libviprs' selection
            // has to agree channel for channel.
            (
                "rgba_aov_half_zip",
                "R,G,B,A",
                "6aeb8bf8858b1fcbd4e99c752dd75fe3bbee77feaacb38fe5f25b31f4890afd7",
            ),
            // The black-image wart, reproduced exactly through the funnel
            // and NOT in the raster libviprs actually returns.
            (
                "z_float_zip",
                "Z",
                "c1538d4cd57a608431e67d7ed8be9460673d60751dfedc5d75a2c9bd5976ca68",
            ),
            // FLOAT samples. The funnel's half rounding is the entire
            // difference: apply it and the digests agree.
            (
                "rgba_float_fine",
                "R,G,B,A",
                "3051352d60e56288f5729ad9ac39acff67317bd5f05f5bd863dc0ec914815419",
            ),
        ];

        for (name, channels, want) in cases {
            let raster = decode_exr(&fixture(name), DecodeLimits::default())
                .unwrap_or_else(|e| panic!("{name} must decode: {e}"));
            assert_eq!(
                raster.get_field("exr-channels"),
                Some(MetadataValue::Str(channels.to_owned())),
                "{name}: the funnel below is written for this exact selection"
            );
            let projected = through_the_vips_rgba_funnel(&raster, channels);
            let got = crate::hex::hex_lower(&Sha256::digest(&projected));
            assert_eq!(
                got, want,
                "{name}: projecting the libviprs decode through vips's RGBA-half \
                 funnel must reproduce the captured `vips rawsave` payload byte \
                 for byte"
            );
        }
    }

    /**
     * Tests that every seed in the fuzz corpus produces exactly the
     * outcome its filename promises, so a regression in one of them is a
     * `cargo test` failure rather than something only a fuzz run would
     * notice.
     * Works by decoding each file under `fuzz/corpus/fuzz_exr/` under the
     * same small allocation budget the fuzz target uses and checking the
     * result against the name. The `data-window-bomb` seed is the shape
     * that matters most: 455 bytes declaring a 200000x200000 window, which
     * has to be refused from the header rather than allocated and then
     * regretted. `valid-many-channels` is the other budget shape: sixteen
     * declared channels where four are selected, so a mutator that grows
     * the channel list grows the decoder's allocation with it.
     * Input: twelve corpus files -> Output: the named outcome from each,
     * and no panic from any of them.
     */
    #[test]
    fn the_fuzz_corpus_decodes_or_fails_exactly_as_named() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("fuzz")
            .join("corpus")
            .join("fuzz_exr");
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
            let result = decode_exr(&bytes, limits);
            seen += 1;

            let ok = match name.as_str() {
                // No magic at all.
                "empty" => matches!(&result, Err(SourceError::Exr(ExrError::BadMagic { .. }))),
                // Magic, then nothing the header parser can use.
                "magic-only" | "truncated-header" | "truncated-body" | "bad-channel-type" => {
                    matches!(&result, Err(SourceError::Exr(ExrError::Decode { .. })))
                }
                // A valid 455-byte file whose data window has been
                // rewritten to 200000x200000. The pixel ceiling catches
                // it before the allocation budget ever comes up.
                "data-window-bomb" => matches!(
                    &result,
                    Err(SourceError::DimensionLimitExceeded {
                        width: 200_000,
                        height: 200_000,
                        ..
                    })
                ),
                // Deep data: one sample list per pixel. vips cannot read
                // this either, it just reports the OpenEXR message.
                "valid-deep-scanline" => {
                    matches!(&result, Err(SourceError::Exr(ExrError::DeepData)))
                }
                // UINT channels, pending the carrier in issue #517.
                "valid-uint-channels" => matches!(
                    &result,
                    Err(SourceError::Exr(ExrError::UnsupportedSampleType { .. }))
                ),
                "valid-rgba-half-zip" | "valid-tiled" => {
                    matches!(&result, Ok(r) if r.format().channels() == 4)
                }
                "valid-depth-only" => matches!(&result, Ok(r) if r.format().channels() == 1),
                // Sixteen declared channels, four selected. Under this
                // seed's 4 MiB budget the 2048-byte declared price passes,
                // so the fuzzer gets a file whose channel list it can grow
                // against the check rather than one that is refused up
                // front.
                "valid-many-channels" => {
                    matches!(&result, Ok(r) if r.format().channels() == 4)
                }
                other => panic!(
                    "corpus file {other:?} has no expected outcome; add one here \
                     when you add a seed"
                ),
            };
            assert!(ok, "corpus file {name:?} gave {result:?}");
        }
        assert_eq!(seen, 12, "the corpus should hold twelve seeds");
    }

    /**
     * Tests that the lossy codings still decode, and that they are NOT
     * pinned as equal to the lossless payload.
     * Works by decoding B44 and asserting it differs from ZIP by the
     * amount vips measured (max absolute difference 4.0), which keeps
     * the lossless assertion above honest: if B44 ever started matching,
     * the fixture would have stopped exercising the lossy path.
     * Input: `rgba_half_b44.exr` -> Output: an 8x4 raster differing from
     * the ZIP payload.
     */
    #[test]
    fn lossy_codings_decode_and_are_not_pinned_as_lossless() {
        let reference = decode_exr(&fixture("rgba_half_zip"), DecodeLimits::default()).unwrap();
        let b44 = decode_exr(&fixture("rgba_half_b44"), DecodeLimits::default()).unwrap();
        assert_eq!((b44.width(), b44.height()), (8, 4));
        assert_ne!(
            b44.data(),
            reference.data(),
            "B44 is lossy by construction, so an exact match here would mean the \
             fixture is no longer exercising it"
        );
        assert_eq!(
            b44.get_field("exr-compression"),
            Some(MetadataValue::Str("b44".to_owned()))
        );
    }
}
