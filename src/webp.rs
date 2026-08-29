//! Still-image WebP (`.webp`) load and lossless save: every RIFF container
//! in, a `VP8L` one out.
//!
//! Ported from libvips `foreign/webp2vips.c` and `foreign/webpsave.c`
//! (v8.18.0-95-gfe420cf3a for the line numbers quoted below; every measured
//! number comes from the 8.18.4 release binary, which is a different
//! artefact). libvips wraps libwebp and so reaches the whole format;
//! libviprs reaches it through `image-webp`, which decodes all of it and
//! encodes exactly one corner of it. That asymmetry is the shape of this
//! module and the reason [`SaveOptions`] looks nothing like `webpsave`'s
//! nineteen options.
//!
//! # Operations
//!
//! | libviprs method | libvips equivalent | result |
//! |---|---|---|
//! | [`decode_webp`] | `webpload` / `webpload_buffer` (default `n = 1`) | 8-bit [`PixelFormat::Rgb8`] or [`PixelFormat::Rgba8`] raster, plus `icc-profile-data` / `exif-data` / `xmp-data` / `n-pages` |
//! | [`decode_webp_with`] | `webpload` with `page` and `n` | the frames asked for, stacked into one toilet-roll raster, plus `page-height` / `delay` / `loop` |
//! | [`Raster::encode_webp`] | `webpsave_buffer --lossless` | `.webp` bytes |
//! | [`Raster::save_webp`] | `webpsave --lossless` | `.webp` file |
//!
//! # Semantics
//!
//! * **The encoder is lossless and nothing else.** `image-webp` 0.2.4's
//!   `encoder.rs` writes a `VP8L` chunk and has no quality knob, no lossy
//!   path, and no `ANIM`/`ANMF` writing anywhere in it. That is why
//!   [`SaveOptions`] carries a [`Compression`] rather than a `quality: u8`:
//!   an argument the encoder throws away inverts the contract (ask for
//!   quality 10, get a lossless file possibly larger than the PNG you
//!   started from) and is a semver time bomb, because the day a lossy
//!   encoder lands every existing `encode_webp(10)` would silently start
//!   emitting small lossy files in a patch release. Making quality
//!   unrepresentable turns that into a compile error today instead, and
//!   [`Compression`] is `#[non_exhaustive]` so `Lossy { .. }` can join it
//!   as a minor bump.
//! * **Not implemented, and not silently accepted either.** Of
//!   `webpsave`'s options, libviprs honours `keep` (as [`Keep`]) and
//!   nothing else. `Q`, `lossless = false`, `preset`, `effort`,
//!   `near-lossless`, `alpha-q`, `min-size`, `target-size`, `kmin`/`kmax`,
//!   `mixed`, `passes`, `smart-subsample`, `smart-deblock` and animated
//!   save have no spelling in [`SaveOptions`] at all. None of them is a
//!   field this module ignores; there is nowhere to write them down.
//! * **Decode is the whole format.** Lossy `VP8`, lossless `VP8L`, alpha,
//!   and the extended `VP8X` container all decode, and the lossy path is
//!   bit-exact against libwebp rather than merely close: VP8
//!   reconstruction is integer-specified and `image-webp` defaults to the
//!   same fancy (bilinear) chroma upsampling, so the pins in the tests
//!   carry no tolerance.
//! * **An animation loads frame 0 by default and every frame on
//!   request.** A default `vips webpload` reads one frame and sets
//!   `n-pages` to the count in the *original* (`webp2vips.c:505-508`);
//!   [`decode_webp`] matches that default exactly and
//!   [`decode_webp_with`] takes the `page` and `n` that ask for more,
//!   stacking the frames into the toilet-roll layout
//!   [`crate::frames`] describes (issue #569).
//! * **Animated WebP can be read and never written.** No pure-Rust
//!   encoder emits `ANIM`/`ANMF`: `image-webp` 0.2.4 writes one `VP8L`
//!   chunk and has no animation surface at all, so [`SaveOptions`] has
//!   nowhere to spell a frame delay or a loop count. So a roll loaded by
//!   [`decode_webp_with`] and handed straight back to
//!   [`Raster::encode_webp`] comes out as **one tall still image**, four
//!   pages deep, and `vips webpsave` on the same raster would have written
//!   a four-frame animation. That is a real divergence and it is pinned
//!   rather than assumed (`an_animation_saved_back_is_one_tall_still`).
//!
//!   Refusing a paged raster was the alternative and it is worse: the
//!   pixels are a perfectly good image, the crate has no other way to spell
//!   "save this roll", and a refusal would fire on the ordinary path of
//!   loading two pages and saving the result. A caller who wants one frame
//!   uses [`Raster::try_extract_page`], and a caller who wants an animation
//!   saves GIF, which is the one animated format in this crate with a
//!   pure-Rust encoder behind it.
//! * **16-bit input is refused, where vips narrows it.** `webpsave`
//!   accepts a `ushort` image and right-shifts it by 8 on the way in
//!   (measured: 255 becomes 0, 256 becomes 1, 65535 becomes 255, and the
//!   same whether the image is tagged `rgb16` or `srgb`).
//!   [`Raster::encode_webp`] returns an error naming the remedy instead.
//!   The reason is internal consistency rather than taste:
//!   [`Raster::cast`](crate::Raster::cast) to an 8-bit format *clips*, so
//!   an automatic narrow here would disagree with the crate's own cast
//!   while looking like it did the same thing. Letting the caller choose
//!   keeps one narrowing rule in the crate, and matches the float refusal
//!   [`crate::sink::encode_png`] and [`Raster::encode_radiance`] already
//!   make. A `narrow` field can join [`SaveOptions`] later without a
//!   breaking change if the parity matters more than the surprise.
//! * **One band becomes three on the round trip.** WebP has no greyscale:
//!   `vips -l` registers the saver as `rgb alpha`, and a `b-w` uchar image
//!   saved and reloaded reports `3 bands, srgb`. libviprs hands `L8` to
//!   the encoder, which is a compression hint rather than a stored colour
//!   type, and the file reads back as [`PixelFormat::Rgb8`] with the
//!   luminance repeated. Alpha survives as a fourth band.
//! * **16383 is the width and height ceiling, not 16384.**
//!   `webpsave.c:740-742` guards on `> 16383` with `image too large`,
//!   which is libwebp's `WEBP_MAX_DIMENSION`. `image-webp` 0.2.4's
//!   `encode_frame` guards on `> 16384` instead, so it will write a
//!   16384-wide `VP8L` that the reference decoder refuses to read.
//!   [`Raster::encode_webp`] applies the libwebp ceiling.
//! * **Metadata travels under the JPEG loader's field names.**
//!   `webp2vips.c:393-397` maps `ICCP`, `EXIF` and `XMP ` to
//!   `icc-profile-data`, `exif-data` and `xmp-data`, as raw chunk payloads
//!   with nothing stripped: a WebP `EXIF` chunk has no `Exif\0\0` prefix
//!   to remove, unlike a JPEG APP1 segment. Save writes back only what is
//!   attached, where vips additionally *synthesises* an EXIF block from
//!   the resolution fields, so a libviprs-written file has no `EXIF` chunk
//!   when the raster carried none.
//!
//! Every number this module is pinned against was measured on the real
//! vips 8.18.4 binary and is recorded, with the commands that produced it,
//! in `oracle-captures/foreign-webp/`.
//!
//! Every entry point here is fallible and there is no panicking twin,
//! matching the rest of the codec surface in [`crate::encode`],
//! [`crate::radiance`] and [`crate::gif`]: a decoder's failures come from
//! untrusted bytes, so a panicking spelling would have no honest caller.

use std::borrow::Cow;
use std::io::Cursor;
use std::path::Path;

use crate::codec::EncodeError;
use crate::frames::{FrameDelay, LoopCount};
use crate::imageio::{MetadataValue, SaveError};
use crate::pixel::PixelFormat;
use crate::raster::Raster;
use crate::source::{DeclaredGeometry, DecodeLimits, SourceError, resolve_page_range};

/// The largest width or height libwebp will encode
/// (`WEBP_MAX_DIMENSION`; `webpsave.c:740-742` rejects anything above it
/// with `image too large`).
///
/// `image-webp` 0.2.4 guards on `> 16384` rather than `> 16383`, one
/// pixel too generous, so a raster of exactly 16384 would encode into a
/// `VP8L` the reference decoder then refuses. libviprs applies this
/// ceiling instead of the crate's.
pub const MAX_DIMENSION: u32 = 16383;

/// The `ANMF` frame-info bit that turns alpha blending **off** for that
/// frame (WebP container spec: the byte after the 24-bit frame duration,
/// where bit 1 set means "do not blend").
///
/// libviprs sets it on frames that provably carry no transparency, and
/// [`disable_blending_on_opaque_frames`] is where and why.
const ANMF_NO_BLEND: u8 = 0b10;

/// The `VP8L` signature byte, which every lossless bitstream opens with.
const VP8L_SIGNATURE: u8 = 0x2f;

/// The RIFF chunks libvips lifts into image metadata, paired with the
/// field name it uses (`vips__webp_names`, `webp2vips.c:393-397`).
///
/// The pairing is the whole reason a WebP profile is readable by the same
/// `raster.icc_profile()` call a JPEG one is.
const METADATA_FIELDS: [(&str, &str); 3] = [
    ("ICCP", "icc-profile-data"),
    ("EXIF", "exif-data"),
    ("XMP ", "xmp-data"),
];

/// How the WebP encoder compresses pixels (libvips `webpsave`'s `lossless`
/// flag plus its `Q` factor, folded into one axis).
///
/// Lossless is the only representable mode because it is the only mode this
/// build can encode. `Lossy { quality }` joins the enum when there is a
/// lossy encoder to back it, which is a minor bump rather than a breaking
/// one thanks to `#[non_exhaustive]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Compression {
    /// Lossless compression (libvips `webpsave --lossless`). The pixels
    /// round-trip exactly and there is no quality factor to set.
    #[default]
    Lossless,
}

/// Which attached metadata [`Raster::encode_webp`] copies into the file
/// (libvips `webpsave`'s `keep`, `VipsForeignKeep`).
///
/// libvips spells this as a flag set with six members. Only the two ends
/// of it are representable here, because the three chunks libviprs can
/// write are exactly the three `keep` distinguishes among the ones WebP
/// has a container slot for. `#[non_exhaustive]` leaves room for the
/// per-chunk members to land as a minor bump if a caller ever needs to
/// keep the profile while dropping the EXIF.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Keep {
    /// Write every attached `icc-profile-data`, `exif-data` and `xmp-data`
    /// blob into its RIFF chunk (libvips `--keep all`, the default there
    /// and here).
    #[default]
    All,
    /// Write pixels and geometry only (libvips `--keep none`, and what
    /// [`Raster::save_stripped`] asks for).
    None,
}

/// Options for [`Raster::encode_webp`] (libvips `webpsave` / `webpsave_buffer`).
///
/// `#[non_exhaustive]`, `Default`, and module-scoped, the same shape as
/// [`DecodeLimits`]: start from
/// [`SaveOptions::default`] and set what you need with the `with_*` builders,
/// e.g. `webp::SaveOptions::default().with_keep(webp::Keep::None)`. That is
/// what makes "later fields can be added without a breaking change" true
/// rather than merely written down (issue #630).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct SaveOptions {
    /// How to compress. Defaults to [`Compression::Lossless`], the only
    /// mode with an encoder behind it.
    pub compression: Compression,
    /// Which attached metadata to carry into the file. Defaults to
    /// [`Keep::All`], as `webpsave`'s `keep` does.
    pub keep: Keep,
}

impl SaveOptions {
    /// Set the compression mode, returning the updated options.
    #[must_use]
    pub fn with_compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    /// Set which attached metadata is carried into the file, returning the
    /// updated options.
    #[must_use]
    pub fn with_keep(mut self, keep: Keep) -> Self {
        self.keep = keep;
        self
    }
}

/// Which frames [`decode_webp_with`] reads out of an animation (libvips
/// `webpload`'s `page` and `n`).
///
/// `#[non_exhaustive]`, `Default`, and module-scoped, the same shape as
/// [`SaveOptions`], [`crate::gif::LoadOptions`] and [`DecodeLimits`]: start
/// from [`LoadOptions::default`] and set what you need with the `with_*`
/// builders, e.g. `webp::LoadOptions::default().with_n(-1)` (issue #630).
///
/// The default is vips's: page 0, one page, so [`decode_webp`] is
/// `decode_webp_with(bytes, limits, LoadOptions::default())` and a still load
/// is unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub struct LoadOptions {
    /// The first frame to load, counting from **zero**, matching vips's
    /// `page` and [`crate::decode_tiff_page`]'s convention (issue #566).
    /// Defaults to 0.
    pub page: u32,
    /// How many frames to load, `-1` for every frame from [`page`](Self::page)
    /// to the end. Defaults to 1, as vips does.
    ///
    /// An `i32` rather than an `Option<u32>` because that is the shape vips's
    /// argument has, sentinel included, and because
    /// [`crate::gif::LoadOptions`] landed with the same field: three sibling
    /// loaders spelling one libvips argument two ways is worse than carrying
    /// its sentinel. Every value the file cannot serve, `0` and `-2` included,
    /// is refused with [`SourceError::PageOutOfRange`], which is what vips
    /// does too.
    pub n: i32,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self { page: 0, n: 1 }
    }
}

impl LoadOptions {
    /// Set the first page to load, returning the updated options.
    #[must_use]
    pub fn with_page(mut self, page: u32) -> Self {
        self.page = page;
        self
    }

    /// Set how many pages to load, `-1` for every remaining page, returning
    /// the updated options.
    #[must_use]
    pub fn with_n(mut self, n: i32) -> Self {
        self.n = n;
        self
    }
}

/// Decode WebP bytes into an 8-bit [`Raster`] (libvips `webpload_buffer`
/// at its default `n = 1`).
///
/// The result is [`PixelFormat::Rgba8`] when the container declares an
/// alpha channel and [`PixelFormat::Rgb8`] otherwise, which is the only
/// pair WebP can hold: there is no greyscale and no 16-bit sample in the
/// format. Lossy `VP8` and lossless `VP8L` both decode, and so does the
/// extended `VP8X` container.
///
/// `ICCP`, `EXIF` and `XMP ` are lifted onto the raster as
/// `icc-profile-data`, `exif-data` and `xmp-data`, the same names the JPEG
/// loader uses, so [`Raster::icc_profile`] finds a WebP profile without
/// knowing where it came from. The payloads are the raw chunk contents.
///
/// # Animations
///
/// An animated file decodes to **frame 0 only**, at one frame's size, and
/// carries `n-pages` set to the number of frames the original had — which
/// is what a default `vips webpload` does (`webp2vips.c:505-508`).
/// [`decode_webp_with`] is the same loader with the `page` and `n` that
/// read more than one.
///
/// [`Raster::get_n_pages`] reads it back for anything under 10,000 frames.
/// At or above that it reports `1`, because it ports
/// `vips_image_get_n_pages`'s sanity ceiling whole (issue #635). Nothing
/// here caps the count on the way in, so an animation that long attaches
/// its real length and the raw value stays readable through
/// [`Raster::get_field`].
///
/// # Errors
///
/// * [`SourceError::Io`] when the container is truncated mid-chunk.
/// * [`SourceError::Decode`] wrapping the codec's own error for a
///   malformed bitstream or a missing chunk, or `image`'s
///   [`LimitErrorKind::InsufficientMemory`](image::error::LimitErrorKind)
///   when the frame buffer or a metadata chunk would exceed
///   [`DecodeLimits::max_alloc_bytes`].
/// * [`SourceError::CoordLimitExceeded`] when either declared axis exceeds
///   [`DecodeLimits::max_coord`].
/// * [`SourceError::DimensionLimitExceeded`] when `width * height` exceeds
///   [`DecodeLimits::max_pixels`], or when the frame buffer the geometry
///   implies does not fit in a `usize`.
/// * [`SourceError::Raster`] when the decoded frame cannot be wrapped
///   (a zero-sized canvas).
pub fn decode_webp(bytes: &[u8], limits: DecodeLimits) -> Result<Raster, SourceError> {
    decode_webp_with(bytes, limits, LoadOptions::default())
}

/// Decode WebP bytes, choosing which frames of an animation to read
/// (libvips `webpload_buffer` with `page` and `n`).
///
/// `options.page` is the first frame, zero-based, and `options.n` is how
/// many to read from there, `-1` meaning every one to the end of the
/// file. Frames are **composited** by the decoder (disposal and blending
/// applied), so each one is a full-size page, and the pages are stacked
/// top to bottom into a single raster in the toilet-roll layout
/// [`crate::frames`] describes. [`decode_webp`] is this function at
/// [`LoadOptions::default`].
///
/// # What comes back attached
///
/// On an animation, and on an animation only:
///
/// * `page-height`, through [`Raster::try_set_page_height`], **when more
///   than one page was loaded**. A one-page load carries no such field,
///   which is what vips does: measured on 8.18.6, `vipsheader -f
///   page-height 'anim.webp[page=1]'` reports the field is not there and
///   `[n=2]` reports 3.
/// * `n-pages`, the count of frames in the **file**, not the count
///   loaded. [`Raster::pages_loaded`] is the second number and
///   [`Raster::get_n_pages`] is this one (issue #635).
/// * `delay`, a [`MetadataValue::IntArray`] of **milliseconds**, one entry
///   per *loaded* page. The `ANMF` duration is milliseconds on the wire so
///   nothing is converted, which is the difference from GIF's
///   centiseconds that [`crate::frames::FrameDelay`] exists to keep
///   visible.
/// * `loop`, the number of plays, `0` meaning forever. The `ANIM` chunk
///   holds the play count with no shift, where GIF's NETSCAPE block counts
///   repeats (measured: `loop = 3` wrote `loop_count = 3`).
/// * `gif-delay` and `gif-loop`, the two compatibility fields
///   `webp2vips.c` attaches beside them: the first delay in centiseconds
///   rounded half to even, and the play count less one, floored at zero.
///
/// A still image gets none of them, exactly as `vipsheader -a` on a still
/// WebP lists none of them.
///
/// # Where this diverges from vips, deliberately
///
/// **The delay array is subset to the pages actually loaded.** vips
/// attaches the file's whole array whatever it loaded: measured,
/// `vipsheader -f delay 'anim4.webp[page=1,n=2]'` prints `45 67 200 12`
/// onto a raster holding pages 1 and 2. Nothing on that raster records the
/// offset, so the array cannot be lined up with the pages that are there,
/// and a saver reading it writes 45 and 67 onto frames that are really 1
/// and 2. Here `delay[i]` is the delay of loaded page `i`, so
/// `delay.len() == pages_loaded()` always holds and the array is usable on
/// its own. `n-pages` stays the file's count, because that one *is*
/// readable without an offset.
///
/// # Errors
///
/// As [`decode_webp`], plus [`SourceError::PageOutOfRange`] when `page` is
/// past the last frame, when `page + n` runs off the end, or when `n` is
/// `0` or below `-1`. vips refuses all of them the same way, with `webp:
/// bad page number`, and clamps none of them.
///
/// The [`DecodeLimits`] ceilings are checked against the **roll**: a
/// four-frame load of a 4x3 animation is priced as 4x12, so `max_coord`,
/// `max_pixels` and `max_alloc_bytes` all see the buffer that will
/// actually be allocated.
pub fn decode_webp_with(
    bytes: &[u8],
    limits: DecodeLimits,
    options: LoadOptions,
) -> Result<Raster, SourceError> {
    // Rewrite the blend flag of any frame that provably carries no
    // transparency before the decoder sees it; the function says why, and
    // it borrows rather than copies when there is nothing to rewrite.
    let bytes = disable_blending_on_opaque_frames(bytes);
    let mut decoder =
        image_webp::WebPDecoder::new(Cursor::new(bytes.as_ref())).map_err(decode_error)?;
    // Budget the metadata chunk reads before any of them run: `read_chunk`
    // refuses a chunk longer than this rather than allocating for it.
    decoder.set_memory_limit(usize::try_from(limits.max_alloc_bytes).unwrap_or(usize::MAX));

    // `dimensions` is the *canvas*, which is one frame of an animation:
    // `image-webp` composites disposal and blending onto it, so every
    // frame it hands back is already a full-size page and the roll is a
    // whole number of them.
    let (width, page_height) = decoder.dimensions();
    let animated = decoder.is_animated();
    // A still image is a one-page file, so the same request resolves
    // against it and `page = 1` is refused there too, exactly as vips
    // refuses `still.webp[page=1]`.
    let file_pages = if animated { decoder.num_frames() } else { 1 };
    let pages = resolve_page_range("webp", options.page, options.n, file_pages)?;
    let loaded = pages.end - pages.start;

    // The ceilings are the roll's, not one frame's. Widened to `u64`
    // before multiplying because a file may declare more frames than the
    // product fits: saturating at `u32::MAX` is safe here only because
    // every ceiling this feeds is far below it, so a saturated height is
    // refused whatever the true one was.
    let height = u32::try_from(u64::from(page_height) * u64::from(loaded)).unwrap_or(u32::MAX);
    // Both ceilings are checked on the declared header geometry, before
    // the frame buffer is reserved, exactly as the shared `image`-crate
    // path in `crate::source` does.
    limits.check_coord(width, height)?;
    limits.check_pixels(width, height)?;

    let format = if decoder.has_alpha() {
        PixelFormat::Rgba8
    } else {
        PixelFormat::Rgb8
    };
    let frame_size = decoder
        .output_buffer_size()
        .ok_or(SourceError::DimensionLimitExceeded {
            width,
            height,
            max_pixels: limits.max_pixels,
        })?;
    let size =
        frame_size
            .checked_mul(loaded as usize)
            .ok_or(SourceError::DimensionLimitExceeded {
                width,
                height,
                max_pixels: limits.max_pixels,
            })?;
    // And the allocation budget, which `check_pixels` does not imply: a
    // 1-gigapixel `max_pixels` permits a 4 GiB `Rgba8` frame, four times the
    // default `max_alloc_bytes`.
    //
    // This used to fabricate an `image::ImageError::Limits` so it reported the
    // same shape as the three formats `image` refuses from inside its own
    // decoder. That was the wrong thing to be consistent with (issue #686).
    // The frames refused are set by the comparison, not by the error type, so
    // reporting through `image` changed nothing about *which* files come back
    // and only threw away the geometry and the price libviprs already had in
    // hand. JPEG, PNG and single-image TIFF genuinely have neither, because
    // the ceiling is spent inside `image` through `Limits::reserve`; WebP has
    // both, on these two lines.
    //
    // The price is the decoder's `output_buffer_size` times the pages asked
    // for rather than a declared product, so it goes through `check_alloc`
    // with the geometry attached by hand rather than through
    // `check_image_alloc`, which would recompute it.
    if limits.exceeds_alloc_budget(size as u64) {
        return Err(SourceError::AllocLimitExceeded {
            what: "WebP frame buffer",
            geometry: Some(DeclaredGeometry {
                width,
                height,
                bands: format.channels() as u32,
            }),
            needed_bytes: size as u64,
            max_alloc_bytes: limits.max_alloc_bytes,
        });
    }

    // Metadata first, so an over-budget chunk is reported before the frame
    // allocation rather than after it.
    let chunks: Vec<(&str, Option<Vec<u8>>)> = METADATA_FIELDS
        .iter()
        .map(|(chunk, field)| {
            let blob = match *chunk {
                "ICCP" => decoder.icc_profile(),
                "EXIF" => decoder.exif_metadata(),
                _ => decoder.xmp_metadata(),
            };
            blob.map(|b| (*field, b)).map_err(decode_error)
        })
        .collect::<Result<_, _>>()?;

    let mut data = vec![0u8; size];
    let mut delays: Vec<i64> = Vec::with_capacity(loaded as usize);
    if animated {
        // `read_frame` reads forward and has no seek, so a `page` past the
        // first costs the frames before it in decode time. They are read
        // into the roll's first slot and overwritten by page `page`
        // itself, which is why this loop does not need a scratch frame.
        for index in 0..pages.end {
            let slot = index.saturating_sub(pages.start) as usize;
            let offset = slot * frame_size;
            let duration = decoder
                .read_frame(&mut data[offset..offset + frame_size])
                .map_err(decode_error)?;
            if index >= pages.start {
                // Milliseconds on the wire, milliseconds in the field, and
                // the type says so: the `ANMF` duration needs no
                // conversion where the GIF loader's centiseconds do.
                delays.push(i64::from(FrameDelay::from_millis(duration).millis()));
            }
        }
    } else {
        decoder.read_image(&mut data).map_err(decode_error)?;
    }

    let mut raster = Raster::new(width, height, format, data)?;
    for (field, blob) in chunks {
        if let Some(blob) = blob {
            raster.fields.set(field, MetadataValue::Blob(blob));
        }
    }
    if animated {
        // The *file's* page count, which is not the raster's: measured,
        // `vipsheader 'anim4.webp[page=1,n=2]'` reports `n-pages: 4` on a
        // raster holding two (issue #635). `Raster::pages_loaded` is the
        // other number.
        raster.set_n_pages(file_pages);
        if loaded > 1 {
            // Only when there is more than one page. Measured: a default
            // load and a `page=1` load both come back with no
            // `page-height` field at all, and only `n > 1` attaches one.
            // The setter refuses a height that does not divide the roll,
            // so a miscount here fails loudly rather than writing a split
            // the reader would discard.
            raster.try_set_page_height(page_height)?;
        }
        let plays = match decoder.loop_count() {
            image_webp::LoopCount::Forever => LoopCount::FOREVER,
            image_webp::LoopCount::Times(times) => LoopCount::from_webp_wire(times.get()),
        };
        raster
            .fields
            .set("loop", MetadataValue::Int(i64::from(plays.plays())));
        // The two compatibility fields `webp2vips.c` attaches beside the
        // real ones, measured on 8.18.6 rather than assumed: `gif-delay`
        // is the first delay in centiseconds under the same
        // round-half-to-even the GIF wire uses (45 ms gives 4, not 5), and
        // `gif-loop` counts repeats after the first play, so 3 plays gives
        // 2 and both `loop 0` and `loop 1` give 0.
        raster.fields.set(
            "gif-loop",
            MetadataValue::Int(i64::from(plays.to_gif_wire().unwrap_or(0))),
        );
        if let Some(&first) = delays.first() {
            let centiseconds = FrameDelay::from_millis(first as u32).to_centiseconds();
            raster
                .fields
                .set("gif-delay", MetadataValue::Int(i64::from(centiseconds)));
        }
        // Subset to the pages actually loaded, which is where this loader
        // parts company with vips; the reason is in the `# Animations`
        // section of the entry point above.
        raster.fields.set("delay", MetadataValue::IntArray(delays));
    }
    Ok(raster)
}

impl Raster {
    /// Encode as lossless WebP bytes (libvips `webpsave_buffer --lossless`).
    ///
    /// Accepts [`PixelFormat::Gray8`], [`PixelFormat::Rgb8`] and
    /// [`PixelFormat::Rgba8`]. Those are the only three the format has a
    /// spelling for, and greyscale is not really one of them: WebP stores
    /// no mono, so `Gray8` is passed as an encoder hint and reads back as
    /// three equal bands, which is what `vips webpsave` does with a `b-w`
    /// image too.
    ///
    /// With [`Keep::All`] the attached `icc-profile-data`, `exif-data` and
    /// `xmp-data` blobs are written into their `ICCP`, `EXIF` and `XMP `
    /// chunks, which promotes the file to the extended `VP8X` container.
    /// With no metadata attached, or under [`Keep::None`], the output is
    /// the simple `RIFF`/`WEBP`/`VP8L` form. Unlike `webpsave`, nothing is
    /// synthesised: a raster carrying no EXIF produces a file with no
    /// `EXIF` chunk, where vips would manufacture one from `xres`/`yres`.
    ///
    /// # Errors
    ///
    /// [`EncodeError::Encode`] when the raster is 16-bit, float, or
    /// multiband (cast to `Gray8`/`Rgb8`/`Rgba8` first — the message says
    /// so), when either axis exceeds [`MAX_DIMENSION`], or when the codec
    /// rejects the frame.
    pub fn encode_webp(&self, options: SaveOptions) -> Result<Vec<u8>, EncodeError> {
        let SaveOptions { compression, keep } = options;
        // One arm today. The `match` is deliberate: when `Lossy` lands it
        // will fail to compile here rather than silently encode losslessly.
        let Compression::Lossless = compression;

        let color = encoder_color_type(self.format())?;
        let (width, height) = (self.width(), self.height());
        if width > MAX_DIMENSION || height > MAX_DIMENSION {
            return Err(EncodeError::encode(format!(
                "webp cannot hold a {width}x{height} image; libwebp's ceiling \
                 is {MAX_DIMENSION} on each axis and vips reports `image too large` past it"
            )));
        }

        let mut out = Vec::new();
        let mut encoder = image_webp::WebPEncoder::new(&mut out);
        if keep == Keep::All {
            for (chunk, field) in METADATA_FIELDS {
                let Some(MetadataValue::Blob(blob)) = self.fields.get(field) else {
                    continue;
                };
                match chunk {
                    "ICCP" => encoder.set_icc_profile(blob.clone()),
                    "EXIF" => encoder.set_exif_metadata(blob.clone()),
                    _ => encoder.set_xmp_metadata(blob.clone()),
                }
            }
        }
        // `encode` asserts on a buffer that does not match the geometry.
        // `Raster` guarantees `data().len() == width * height * bpp` by
        // construction, and the debug assertion says so out loud rather
        // than leaving the panic to the dependency.
        debug_assert_eq!(
            self.data().len(),
            self.stride() * height as usize,
            "a Raster's buffer is exactly its geometry"
        );
        encoder
            .encode(self.data(), width, height, color)
            .map_err(EncodeError::encode)?;
        Ok(out)
    }

    /// Save the raster to `path` as lossless WebP (libvips `webpsave`).
    ///
    /// # Errors
    ///
    /// [`SaveError::Encode`] when [`Raster::encode_webp`] rejects the
    /// raster, or [`SaveError::Io`] when the file write fails.
    pub fn save_webp(&self, path: &Path, options: SaveOptions) -> Result<(), SaveError> {
        let bytes = self.encode_webp(options).map_err(|e| match e {
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

/// The `.webp` row of [`Raster::save`]'s extension route
/// (`crate::imageio::save_impl`).
///
/// Exists so the extension route carries one call rather than the
/// keep-flag translation and the error mapping, and so both live next to
/// the encoder they belong to. `keep_metadata` is the flag that separates
/// [`Raster::save`] from [`Raster::save_stripped`]; it maps onto
/// [`Keep::All`] and [`Keep::None`], which is exactly the `--keep all` /
/// `--keep none` pair `vips webpsave` takes.
pub(crate) fn encode_webp_for_save(
    raster: &Raster,
    keep_metadata: bool,
) -> Result<Vec<u8>, SaveError> {
    let keep = if keep_metadata { Keep::All } else { Keep::None };
    raster
        .encode_webp(SaveOptions {
            keep,
            ..Default::default()
        })
        .map_err(|e| match e {
            EncodeError::Io(io) => SaveError::Io(io),
            other => SaveError::Encode(crate::sink::SinkError::EncodeMsg(other.to_string())),
        })
}

/// Where the chunk after this one starts, given the offset its payload
/// starts at and the size its header declares, or `None` when the file's own
/// arithmetic runs off the end of the address space.
///
/// RIFF pads an odd payload to an even length, so the step is `size + (size
/// & 1)`, and **both** additions have to be checked rather than only the
/// outer one. `size` comes straight off the wire as a `u32`, and on a 32-bit
/// target `u32::MAX as usize` *is* `usize::MAX`, so a chunk declaring the
/// largest size its four-byte field can hold overflows on the pad, before
/// the outer addition ever sees it: a panic with overflow checks on, and a
/// wrapped zero with them off, which is a walk that never advances.
///
/// This is a free function rather than two lines inline because the case
/// that reaches it cannot be built on a 64-bit host, where the same file
/// gives a `Some` far past the end of the buffer and the walk stops on the
/// next `get`. Hoisting it makes the overflow reachable from a test on any
/// target (issue #862).
fn next_chunk(payload: usize, size: usize) -> Option<usize> {
    // Only the outer addition is checked, which is issue #862 itself: the
    // pad has already overflowed by the time this looks. The next commit is
    // the fix.
    payload.checked_add(size + (size & 1))
}

/// Byte offsets of the `ANMF` frame-info bytes whose frame provably
/// carries no transparency **and** asks to be alpha-blended anyway.
///
/// Walks the top-level RIFF chunk chain and reads each animation frame's
/// own sub-chunk to decide, because that is where the answer is: a `VP8 `
/// frame is lossy and has no alpha channel at all, a `VP8L` frame declares
/// one in its `alpha_is_used` header bit, and an `ALPH` frame has one by
/// construction. Anything it cannot parse is left alone.
///
/// Nothing is written here; see [`disable_blending_on_opaque_frames`] for
/// why the offsets are wanted.
fn opaque_blended_frame_offsets(bytes: &[u8]) -> Vec<usize> {
    let mut offsets = Vec::new();
    if bytes.len() < 12 || &bytes[..4] != b"RIFF" || &bytes[8..12] != b"WEBP" {
        return offsets;
    }
    let mut cursor = 12usize;
    while let Some(header) = bytes.get(cursor..cursor + 8) {
        let size = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
        let payload = cursor + 8;
        // An `ANMF` payload is a 16-byte frame header (x, y, width, height
        // and duration as 24-bit fields, then the frame-info byte) and then
        // one sub-chunk, whose own 8-byte header has to be inside it too.
        if &header[..4] == b"ANMF"
            && size >= 24
            && let Some(frame) = bytes.get(payload..payload + size.min(29))
            && frame.len() >= 24
            && frame[15] & ANMF_NO_BLEND == 0
        {
            let opaque = match &frame[16..20] {
                b"VP8 " => true,
                // `VP8L`: signature byte, then 14 bits of width, 14 of
                // height, then `alpha_is_used`, which is bit 28 of the
                // little-endian word that follows the signature.
                b"VP8L" => {
                    frame.len() >= 29
                        && frame[24] == VP8L_SIGNATURE
                        && (u32::from_le_bytes(frame[25..29].try_into().unwrap()) >> 28) & 1 == 0
                }
                _ => false,
            };
            if opaque {
                offsets.push(payload + 15);
            }
        }
        // Chunk payloads are padded to an even length, and the walk stops
        // rather than wrapping on a size a hostile file inflated. Both
        // additions are checked inside `next_chunk`; doing only the outer
        // one here is what issue #862 was.
        cursor = match next_chunk(payload, size) {
            Some(next) => next,
            None => break,
        };
    }
    offsets
}

/// Return `bytes` with alpha blending switched off on every animation
/// frame that provably carries no transparency, borrowing the input
/// unchanged when there is nothing to switch.
///
/// # Why a decoder needs this
///
/// libwebp does **not** blend a source pixel whose alpha is 255; it copies
/// it (`demux/anim_decode.c`, `BlendPixelRowNonPremult`, which tests
/// `src_alpha != 0xff` before calling the blend). `image-webp` 0.2.4 has
/// the same approximate blend arithmetic and no such test, so it runs the
/// approximation on opaque pixels too, and the approximation is not exact
/// there: with `src_a = 255` the scale is `(1 << 24) / 255`, the product is
/// `s * 255 * 65793 = s * 16777215`, and `>> 24` of that is `s - 1` for
/// every `s` from 1 to 255.
///
/// So every opaque pixel of a blended frame comes back one grey level
/// low. Measured on the pinned vips 8.18.6: `vips webpsave` on a four-page
/// opaque roll writes frame 0 with blending off and frames 1, 2 and 3 with
/// blending **on**, `vips rawsave 'x.webp[n=-1]'` reads back the original
/// ramp, and libviprs read `74 20 38` where vips read `75 21 39`, on every
/// channel of every page but the first.
///
/// Blending a fully opaque frame is the identity, so clearing the bit on
/// exactly those frames cannot change the image and does route the decoder
/// onto its exact copy path. The residue is a frame that declares alpha and
/// asks to be blended, where the opaque pixels *inside* it are still one
/// low; `vips webpsave` does not write that combination (a transparent roll
/// comes out with blending off on every frame, measured), so no oracle
/// fixture reaches it, and it is filed rather than hidden.
///
/// The clone is the whole file and it is deliberately not priced against
/// [`DecodeLimits::max_alloc_bytes`]: it is one copy of a buffer the caller
/// already holds rather than an expansion of it, and it only happens for a
/// file that actually carries such a frame.
/// `crate::encode_tiff`'s `normalize_multiband_photometric` makes the same
/// trade for the same reason.
fn disable_blending_on_opaque_frames(bytes: &[u8]) -> Cow<'_, [u8]> {
    let offsets = opaque_blended_frame_offsets(bytes);
    if offsets.is_empty() {
        return Cow::Borrowed(bytes);
    }
    let mut owned = bytes.to_vec();
    for offset in offsets {
        owned[offset] |= ANMF_NO_BLEND;
    }
    Cow::Owned(owned)
}

/// The encoder colour type for a raster, or the reason there is none.
///
/// WebP samples are 8-bit and the container holds one, three, or four of
/// them per pixel. Everything wider is refused rather than narrowed; see
/// the module docs for why libviprs does not copy vips's silent `>> 8`.
fn encoder_color_type(format: PixelFormat) -> Result<image_webp::ColorType, EncodeError> {
    match format {
        PixelFormat::Gray8 => Ok(image_webp::ColorType::L8),
        PixelFormat::Rgb8 => Ok(image_webp::ColorType::Rgb8),
        PixelFormat::Rgba8 => Ok(image_webp::ColorType::Rgba8),
        PixelFormat::Gray16 | PixelFormat::Rgb16 | PixelFormat::Rgba16 => {
            Err(EncodeError::encode(format!(
                "webp samples are 8-bit and {format:?} is 16-bit; cast to an 8-bit \
                 format first, so the narrowing is yours rather than the encoder's \
                 (vips webpsave narrows silently, by a right shift of 8)"
            )))
        }
        other => Err(EncodeError::encode(format!(
            "webp holds 1, 3 or 4 bands of 8-bit samples and {other:?} has no such \
             spelling; cast to Gray8, Rgb8 or Rgba8 first"
        ))),
    }
}

/// Map the codec's decode failure onto the shared decode error.
///
/// An I/O failure keeps its own variant so a truncated file reads as one;
/// everything else is wrapped the way the `image` facade wraps it, so a
/// malformed WebP reports through the same [`SourceError::Decode`] arm it
/// did before this module owned the decode.
fn decode_error(err: image_webp::DecodingError) -> SourceError {
    match err {
        image_webp::DecodingError::IoError(io) => SourceError::Io(io),
        other => SourceError::Decode(image::ImageError::Decoding(
            image::error::DecodingError::new(image::ImageFormat::WebP.into(), other),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::imageio::MetadataValue;
    use crate::pixel::PixelFormat;
    use crate::source::{DecodeLimits, decode_bytes_with_limits};

    // -----------------------------------------------------------------
    // Oracle fixtures. Every byte below came out of vips 8.18.4; the
    // commands are in `oracle-captures/foreign-webp/commands.sh` and the
    // expected pixels in its `oracle.json`.
    // -----------------------------------------------------------------

    /// `vips webpsave --lossless --keep none` on the 4x3 sRGB raster in
    /// [`ramp_rgb`], captured verbatim. One `VP8L` chunk, 100 bytes.
    const LOSSLESS_RGB: [u8; 100] = [
        0x52, 0x49, 0x46, 0x46, 0x5c, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50, 0x56, 0x50, 0x38,
        0x4c, 0x50, 0x00, 0x00, 0x00, 0x2f, 0x03, 0x80, 0x00, 0x00, 0x5f, 0x60, 0x98, 0x8d, 0xa4,
        0x7d, 0x0e, 0xcd, 0x7a, 0x2d, 0xd1, 0x79, 0x1d, 0xcd, 0xa0, 0x18, 0x64, 0x1b, 0xe9, 0x04,
        0x4e, 0xe0, 0x90, 0x06, 0x3c, 0xa4, 0x17, 0x78, 0x01, 0x83, 0x6c, 0x23, 0x1d, 0xc2, 0x44,
        0x4e, 0xe8, 0x01, 0x4f, 0xe8, 0x45, 0x86, 0x30, 0xff, 0x91, 0xa4, 0x48, 0xf4, 0x77, 0x44,
        0xfb, 0xea, 0x06, 0x81, 0xe8, 0x06, 0x71, 0x7d, 0x20, 0x08, 0x20, 0xf1, 0x48, 0x28, 0x80,
        0x0d, 0x26, 0xd9, 0x63, 0xf1, 0x88, 0xfe, 0xc7, 0x82, 0x2f,
    ];

    /// The same 4x3 raster with an alpha ramp, `vips webpsave --lossless
    /// --keep none`. One `VP8L` chunk, 116 bytes.
    const LOSSLESS_RGBA: [u8; 116] = [
        0x52, 0x49, 0x46, 0x46, 0x6c, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50, 0x56, 0x50, 0x38,
        0x4c, 0x60, 0x00, 0x00, 0x00, 0x2f, 0x03, 0x80, 0x00, 0x10, 0x5f, 0xa0, 0xa4, 0x8d, 0x24,
        0x68, 0xe1, 0x3d, 0x2a, 0x0f, 0x8e, 0xfb, 0x83, 0x23, 0xc5, 0x20, 0xdb, 0x48, 0x27, 0x76,
        0x02, 0xd3, 0x78, 0xa5, 0x67, 0x39, 0xb1, 0x09, 0x18, 0x66, 0x23, 0x69, 0x11, 0x56, 0x60,
        0x11, 0x86, 0xe5, 0xfc, 0x7f, 0x87, 0x30, 0x08, 0x6a, 0xda, 0x36, 0x82, 0x4a, 0xf1, 0x3e,
        0x38, 0x99, 0x4a, 0xb2, 0x58, 0x6e, 0x8a, 0x24, 0x12, 0x11, 0x41, 0x52, 0xab, 0xff, 0xff,
        0x91, 0xd4, 0xea, 0xeb, 0x6e, 0xef, 0x0a, 0x89, 0x08, 0x08, 0x02, 0x48, 0x3c, 0xa2, 0x89,
        0x61, 0x85, 0x61, 0x66, 0xda, 0x34, 0xa2, 0xff, 0xe1, 0xe8, 0x03,
    ];

    /// `vips webpsave` at the default `Q` on the same 4x3 raster: a `VP8`
    /// chunk, the lossy bitstream libviprs decodes but cannot write.
    const LOSSY_RGB: [u8; 96] = [
        0x52, 0x49, 0x46, 0x46, 0x58, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50, 0x56, 0x50, 0x38,
        0x20, 0x4c, 0x00, 0x00, 0x00, 0xd0, 0x01, 0x00, 0x9d, 0x01, 0x2a, 0x04, 0x00, 0x03, 0x00,
        0x02, 0x00, 0x34, 0x25, 0xb0, 0x02, 0x74, 0x01, 0x0e, 0xfe, 0x03, 0xc8, 0x00, 0x00, 0xfc,
        0x3c, 0x7e, 0x73, 0xd3, 0xe4, 0x80, 0x52, 0xee, 0x82, 0x37, 0xda, 0xf7, 0x4f, 0xea, 0xd3,
        0xe3, 0xd3, 0xf7, 0xff, 0x5b, 0x8b, 0x76, 0x19, 0xcc, 0xfa, 0x2d, 0xf7, 0xdf, 0xee, 0x72,
        0x65, 0x9b, 0xfe, 0x35, 0x44, 0xe9, 0x04, 0x77, 0xca, 0xd5, 0x96, 0xb8, 0xf9, 0xc9, 0xe2,
        0x39, 0xfa, 0xd7, 0xa8, 0x80, 0x00,
    ];

    /// The `LOSSLESS_RGB` bitstream rewrapped in an extended container
    /// with an `ICCP`, an `EXIF` and an `XMP ` chunk, all three flagged in
    /// `VP8X`. vips reports them as 24, 10 and 37 bytes of binary data.
    const META_RGB: [u8; 214] = [
        0x52, 0x49, 0x46, 0x46, 0xce, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50, 0x56, 0x50, 0x38,
        0x58, 0x0a, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x02, 0x00, 0x00,
        0x49, 0x43, 0x43, 0x50, 0x18, 0x00, 0x00, 0x00, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16,
        0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25,
        0x26, 0x27, 0x56, 0x50, 0x38, 0x4c, 0x50, 0x00, 0x00, 0x00, 0x2f, 0x03, 0x80, 0x00, 0x00,
        0x5f, 0x60, 0x98, 0x8d, 0xa4, 0x7d, 0x0e, 0xcd, 0x7a, 0x2d, 0xd1, 0x79, 0x1d, 0xcd, 0xa0,
        0x18, 0x64, 0x1b, 0xe9, 0x04, 0x4e, 0xe0, 0x90, 0x06, 0x3c, 0xa4, 0x17, 0x78, 0x01, 0x83,
        0x6c, 0x23, 0x1d, 0xc2, 0x44, 0x4e, 0xe8, 0x01, 0x4f, 0xe8, 0x45, 0x86, 0x30, 0xff, 0x91,
        0xa4, 0x48, 0xf4, 0x77, 0x44, 0xfb, 0xea, 0x06, 0x81, 0xe8, 0x06, 0x71, 0x7d, 0x20, 0x08,
        0x20, 0xf1, 0x48, 0x28, 0x80, 0x0d, 0x26, 0xd9, 0x63, 0xf1, 0x88, 0xfe, 0xc7, 0x82, 0x2f,
        0x45, 0x58, 0x49, 0x46, 0x0a, 0x00, 0x00, 0x00, 0x49, 0x49, 0x2a, 0x00, 0x08, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x58, 0x4d, 0x50, 0x20, 0x25, 0x00, 0x00, 0x00, 0x3c, 0x78, 0x3a, 0x78,
        0x6d, 0x70, 0x6d, 0x65, 0x74, 0x61, 0x20, 0x78, 0x6d, 0x6c, 0x6e, 0x73, 0x3a, 0x78, 0x3d,
        0x22, 0x61, 0x64, 0x6f, 0x62, 0x65, 0x3a, 0x6e, 0x73, 0x3a, 0x6d, 0x65, 0x74, 0x61, 0x2f,
        0x22, 0x2f, 0x3e, 0x00,
    ];

    /// `vips webpsave --lossless --page-height 3` on a 4x9 toilet-roll: an
    /// animation of three 4x3 frames, whose frame 0 is the `LOSSLESS_RGB`
    /// image. vips reports `n-pages: 3` and loads 4x3 by default.
    const ANIM3: [u8; 374] = [
        0x52, 0x49, 0x46, 0x46, 0x6e, 0x01, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50, 0x56, 0x50, 0x38,
        0x58, 0x0a, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x02, 0x00, 0x00,
        0x41, 0x4e, 0x49, 0x4d, 0x06, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x41,
        0x4e, 0x4d, 0x46, 0x68, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00,
        0x00, 0x02, 0x00, 0x00, 0x64, 0x00, 0x00, 0x02, 0x56, 0x50, 0x38, 0x4c, 0x50, 0x00, 0x00,
        0x00, 0x2f, 0x03, 0x80, 0x00, 0x00, 0x5f, 0x60, 0x98, 0x8d, 0xa4, 0x7d, 0x0e, 0xcd, 0x7a,
        0x2d, 0xd1, 0x79, 0x1d, 0xcd, 0xa0, 0x18, 0x64, 0x1b, 0xe9, 0x04, 0x4e, 0xe0, 0x90, 0x06,
        0x3c, 0xa4, 0x17, 0x78, 0x01, 0x83, 0x6c, 0x23, 0x1d, 0xc2, 0x44, 0x4e, 0xe8, 0x01, 0x4f,
        0xe8, 0x45, 0x86, 0x30, 0xff, 0x91, 0xa4, 0x48, 0xf4, 0x77, 0x44, 0xfb, 0xea, 0x06, 0x81,
        0xe8, 0x06, 0x71, 0x7d, 0x20, 0x08, 0x20, 0xf1, 0x48, 0x28, 0x80, 0x0d, 0x26, 0xd9, 0x63,
        0xf1, 0x88, 0xfe, 0xc7, 0x82, 0x2f, 0x41, 0x4e, 0x4d, 0x46, 0x64, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x02, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00,
        0x56, 0x50, 0x38, 0x4c, 0x4b, 0x00, 0x00, 0x00, 0x2f, 0x03, 0x80, 0x00, 0x00, 0x5f, 0x40,
        0x16, 0x60, 0xf2, 0x8e, 0x26, 0xdd, 0xbc, 0xa2, 0x08, 0x61, 0x28, 0x64, 0x01, 0x26, 0x04,
        0x09, 0x84, 0x34, 0xe0, 0xe1, 0x74, 0xc6, 0xa1, 0x90, 0x8d, 0x24, 0x48, 0x61, 0xf8, 0x16,
        0x70, 0x79, 0xee, 0x11, 0x19, 0x84, 0xf9, 0x8f, 0x6f, 0xa6, 0x85, 0x40, 0xa4, 0x85, 0x68,
        0xde, 0x1e, 0x22, 0xd1, 0x1e, 0x22, 0x90, 0x05, 0x98, 0xfc, 0x33, 0x18, 0x49, 0x06, 0x99,
        0x25, 0x14, 0x6d, 0x44, 0xff, 0x23, 0xec, 0x15, 0x00, 0x41, 0x4e, 0x4d, 0x46, 0x66, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x02, 0x00, 0x00, 0x64,
        0x00, 0x00, 0x00, 0x56, 0x50, 0x38, 0x4c, 0x4d, 0x00, 0x00, 0x00, 0x2f, 0x03, 0x80, 0x00,
        0x00, 0x5f, 0x60, 0x90, 0x6d, 0x24, 0x80, 0x23, 0x3d, 0xba, 0x77, 0x39, 0x85, 0x97, 0x28,
        0x8a, 0x41, 0xb6, 0x91, 0x90, 0xce, 0xa3, 0x1a, 0xcf, 0xfc, 0x0a, 0x27, 0x70, 0x02, 0x66,
        0x02, 0x26, 0x01, 0xcd, 0x43, 0x16, 0x7f, 0xfa, 0x07, 0x58, 0x03, 0xe7, 0x3f, 0xfe, 0x3f,
        0xb8, 0x2c, 0x00, 0xd6, 0x02, 0x5d, 0xe4, 0x4e, 0x17, 0x60, 0x2d, 0x90, 0x05, 0x98, 0x48,
        0x4c, 0x66, 0x36, 0x49, 0xe4, 0xd4, 0x1d, 0x6d, 0x44, 0xff, 0xe3, 0x35, 0x03, 0x00,
    ];

    /// `vips webpsave --lossless --page-height 3 --strip` on a 4x15
    /// toilet roll of five flat greys: an animation of FIVE 4x3 frames.
    /// vips 8.18.6 reports `n-pages: 5` for it and loads 4x3 by default.
    ///
    /// Five is the point. `ANIM3`'s page count collides with its band
    /// count and with one of its axes, so a loader that put the wrong
    /// number under `n-pages` could pass every assertion made against it
    /// (issue #635).
    const ANIM5: [u8; 294] = [
        0x52, 0x49, 0x46, 0x46, 0x1e, 0x01, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50, 0x56, 0x50, 0x38,
        0x58, 0x0a, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x02, 0x00, 0x00,
        0x41, 0x4e, 0x49, 0x4d, 0x06, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x41,
        0x4e, 0x4d, 0x46, 0x2a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00,
        0x00, 0x02, 0x00, 0x00, 0x64, 0x00, 0x00, 0x02, 0x56, 0x50, 0x38, 0x4c, 0x11, 0x00, 0x00,
        0x00, 0x2f, 0x03, 0x80, 0x00, 0x00, 0x07, 0x50, 0x8a, 0x52, 0x94, 0xa2, 0xff, 0x81, 0x88,
        0xe8, 0x7f, 0x00, 0x00, 0x41, 0x4e, 0x4d, 0x46, 0x2a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x02, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00, 0x56, 0x50,
        0x38, 0x4c, 0x11, 0x00, 0x00, 0x00, 0x2f, 0x03, 0x80, 0x00, 0x00, 0x07, 0x50, 0xa3, 0x1a,
        0xd5, 0xa8, 0xff, 0x81, 0x88, 0xe8, 0x7f, 0x00, 0x00, 0x41, 0x4e, 0x4d, 0x46, 0x2a, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x02, 0x00, 0x00, 0x64,
        0x00, 0x00, 0x00, 0x56, 0x50, 0x38, 0x4c, 0x11, 0x00, 0x00, 0x00, 0x2f, 0x03, 0x80, 0x00,
        0x00, 0x07, 0x50, 0xbc, 0xe2, 0x15, 0xaf, 0xff, 0x81, 0x88, 0xe8, 0x7f, 0x00, 0x00, 0x41,
        0x4e, 0x4d, 0x46, 0x2a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00,
        0x00, 0x02, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00, 0x56, 0x50, 0x38, 0x4c, 0x11, 0x00, 0x00,
        0x00, 0x2f, 0x03, 0x80, 0x00, 0x00, 0x07, 0x50, 0xd5, 0xaa, 0x56, 0xb5, 0xff, 0x81, 0x88,
        0xe8, 0x7f, 0x00, 0x00, 0x41, 0x4e, 0x4d, 0x46, 0x2a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x02, 0x00, 0x00, 0x64, 0x00, 0x00, 0x00, 0x56, 0x50,
        0x38, 0x4c, 0x11, 0x00, 0x00, 0x00, 0x2f, 0x03, 0x80, 0x00, 0x00, 0x07, 0x50, 0xee, 0x72,
        0x97, 0xbb, 0xff, 0x81, 0x88, 0xe8, 0x7f, 0x00, 0x00,
    ];

    /// The 4x3 sRGB ramp every fixture above was written from.
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

    /// The twelve RGB triples `vips getpoint` prints for every fixture
    /// that carries the lossless ramp.
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

    /// Read every pixel of `raster` in raster order as bytes.
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

    /**
     * Tests that a lossless WebP written by vips decodes to exactly the
     * pixels vips reads back out of it, so the VP8L path is pinned to the
     * reference decoder rather than to itself. Works by decoding the
     * 100-byte `--lossless --keep none` capture and comparing every pixel
     * to the `vips getpoint` output recorded beside it.
     * Input: `LOSSLESS_RGB` -> Output: 4x3 `Rgb8`, pixels equal to
     * `RAMP_PIXELS`, and no `n-pages` field, which is what `vipsheader
     * -a` reports for the same file.
     */
    #[test]
    fn lossless_decode_matches_vips_getpoint() {
        let raster = decode_bytes_with_limits(&LOSSLESS_RGB, DecodeLimits::default())
            .expect("the vips lossless capture decodes");
        assert_eq!((raster.width(), raster.height()), (4, 3));
        assert_eq!(raster.format(), PixelFormat::Rgb8);
        assert_eq!(pixels(&raster), RAMP_PIXELS.map(Vec::from).to_vec());
        // A still image carries no `n-pages` at all, as vips reports:
        // `vipsheader -a` on this capture lists no such field.
        assert_eq!(raster.get_field("n-pages"), None);
        assert_eq!(raster.get_n_pages(), 1);
    }

    /**
     * Tests that the lossy VP8 path is bit-exact against libwebp, not
     * merely close: VP8 reconstruction is integer-specified and
     * `image-webp` defaults to the same fancy (bilinear) chroma
     * upsampling libwebp does, so the two agree byte for byte. Works by
     * decoding the default-`Q` capture and comparing to the twelve
     * triples vips printed for the same file.
     * Input: `LOSSY_RGB` -> Output: 4x3 `Rgb8`, pixels exactly the vips
     * values, which differ from the original ramp because the encode was
     * lossy.
     */
    #[test]
    fn lossy_decode_is_bit_exact_against_libwebp() {
        let raster = decode_bytes_with_limits(&LOSSY_RGB, DecodeLimits::default())
            .expect("the vips lossy capture decodes");
        assert_eq!(raster.format(), PixelFormat::Rgb8);
        let expected: [[u8; 3]; 12] = [
            [0, 14, 20],
            [56, 92, 82],
            [160, 153, 112],
            [112, 85, 28],
            [73, 125, 155],
            [166, 197, 214],
            [100, 90, 80],
            [201, 171, 147],
            [26, 66, 144],
            [115, 136, 204],
            [219, 204, 253],
            [160, 126, 165],
        ];
        assert_eq!(pixels(&raster), expected.map(Vec::from).to_vec());
        assert_ne!(pixels(&raster), RAMP_PIXELS.map(Vec::from).to_vec());
    }

    /**
     * Tests that an alpha channel survives the lossless decode as a
     * fourth band rather than being flattened, matching the `4 bands`
     * vips reports for the same file. Works by decoding the RGBA capture
     * and checking the alpha ramp.
     * Input: `LOSSLESS_RGBA` -> Output: 4x3 `Rgba8` whose bytes equal the
     * source raster's.
     */
    #[test]
    fn lossless_alpha_decodes_as_four_bands() {
        let raster = decode_bytes_with_limits(&LOSSLESS_RGBA, DecodeLimits::default())
            .expect("the vips lossless RGBA capture decodes");
        assert_eq!(raster.format(), PixelFormat::Rgba8);
        assert_eq!(raster.data(), ramp_rgba().data());
    }

    /**
     * Tests that the three metadata RIFF chunks are lifted onto the
     * raster under the same field names the JPEG loader uses, so a
     * caller reads `icc-profile-data` regardless of which container the
     * profile arrived in. Works by decoding the hand-built extended
     * container and comparing each blob to the exact chunk payload vips
     * reports the size of.
     * Input: `META_RGB` -> Output: `icc-profile-data` = 24 bytes
     * `0x10..0x27`, `exif-data` = the 10-byte little-endian TIFF header,
     * `xmp-data` = the 37-byte packet, and the pixels unchanged.
     */
    #[test]
    fn decode_attaches_icc_exif_and_xmp_from_the_riff_chunks() {
        let raster = decode_bytes_with_limits(&META_RGB, DecodeLimits::default())
            .expect("the extended container decodes");
        assert_eq!(pixels(&raster), RAMP_PIXELS.map(Vec::from).to_vec());
        let blob = |name: &str| match raster.get_field(name) {
            Some(MetadataValue::Blob(b)) => b.clone(),
            other => panic!("{name} should be a blob, got {other:?}"),
        };
        assert_eq!(
            blob("icc-profile-data"),
            (0x10u8..=0x27).collect::<Vec<u8>>()
        );
        assert_eq!(
            blob("exif-data"),
            b"II*\x00\x08\x00\x00\x00\x00\x00".to_vec()
        );
        assert_eq!(
            blob("xmp-data"),
            b"<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"/>".to_vec()
        );
    }

    /**
     * Tests the multi-frame verdict: an animated WebP loads its first
     * frame and says how many there were, which is exactly what a
     * default `vips webpload` does (`n` defaults to 1). The toilet-roll
     * load lives in issue #569 behind the page model, and refusing the
     * file outright would be a regression, since frame 0 already decoded
     * before this lane. Works by decoding a three-frame capture and
     * checking both the geometry and `n-pages`.
     * Input: `ANIM3` -> Output: 4x3 (not 4x9), pixels equal to frame 0,
     * `get_n_pages() == 3`.
     */
    #[test]
    fn animated_webp_loads_frame_zero_and_reports_the_page_count() {
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
     * decoding a five-frame animation, whose count collides with no other
     * number the raster carries, and by counting the fixture's own `ANMF`
     * chunks rather than trusting the 5 as a literal.
     * Input: `ANIM5` -> Output: 4x3 with 3 bands, and `get_n_pages()` equal
     * to the five `ANMF` chunks in the bytes.
     */
    #[test]
    fn n_pages_counts_the_frames_in_the_file_and_nothing_else() {
        let frames = ANIM5.windows(4).filter(|w| *w == b"ANMF").count();
        assert_eq!(frames, 5, "the fixture is a five-frame animation");

        let raster = decode_bytes_with_limits(&ANIM5, DecodeLimits::default())
            .expect("the five-frame animation decodes");
        assert_eq!((raster.width(), raster.height()), (4, 3));
        assert_eq!(raster.bands(), 3);
        assert_eq!(
            u32::try_from(frames).unwrap(),
            raster.get_n_pages(),
            "n-pages is the frame count of the original, not the band count, \
             not an axis, and not the one page that was loaded"
        );
    }

    /**
     * Tests the property the lossless-only encoder buys: because there
     * is no quantisation step anywhere in the pipeline, decoding what
     * `encode_webp` wrote returns the input bytes exactly, for both the
     * opaque and the alpha carrier. Works by encoding two rasters at the
     * default options and decoding the result back.
     * Input: the 4x3 `Rgb8` and `Rgba8` ramps -> Output: identical
     * dimensions, identical pixel format, byte-identical data.
     */
    #[test]
    fn lossless_encode_decode_is_the_identity() {
        for original in [ramp_rgb(), ramp_rgba()] {
            let bytes = original
                .encode_webp(SaveOptions::default())
                .expect("the lossless encoder accepts an 8-bit raster");
            let back = decode_bytes_with_limits(&bytes, DecodeLimits::default())
                .expect("our own bytes decode");
            assert_eq!((back.width(), back.height()), (4, 3));
            assert_eq!(back.format(), original.format());
            assert_eq!(back.data(), original.data(), "lossless is exact");
        }
    }

    /**
     * Tests that a 16-bit raster is refused with a message naming the
     * remedy rather than silently narrowed. vips narrows instead: a
     * `ushort` image saved through `webpsave` comes back right-shifted
     * by 8 (measured: 255 -> 0, 256 -> 1, 511 -> 1, 512 -> 2, 65535 ->
     * 255), which throws away the low byte without telling anyone.
     * libviprs makes the caller pick the narrowing. Works by encoding an
     * `Rgb16` raster and matching the typed error.
     * Input: 4x3 `Rgb16` -> Output: `EncodeError::Encode` whose message
     * names the format and says to cast.
     */
    #[test]
    fn sixteen_bit_is_refused_rather_than_narrowed() {
        let wide = Raster::zeroed(4, 3, PixelFormat::Rgb16).unwrap();
        let err = wide
            .encode_webp(SaveOptions::default())
            .expect_err("WebP has no 16-bit sample spelling");
        let msg = err.to_string();
        assert!(matches!(err, EncodeError::Encode(_)), "{err:?}");
        assert!(msg.contains("Rgb16"), "{msg}");
        assert!(msg.contains("cast"), "{msg}");
    }

    /**
     * Tests that a one-band raster is promoted to three bands on the
     * round trip, because WebP stores no greyscale: `vips webpsave` on a
     * `b-w` uchar image also reports `3 bands, srgb` when it is loaded
     * back. Works by encoding a `Gray8` ramp and checking the decoded
     * bands repeat the luminance.
     * Input: 4x3 `Gray8` -> Output: 4x3 `Rgb8` whose three bands each
     * equal the source luminance.
     */
    #[test]
    fn grey_promotes_to_rgb_on_the_round_trip_as_vips_does() {
        let data: Vec<u8> = RAMP_PIXELS.iter().map(|p| p[0]).collect();
        let grey = Raster::new(4, 3, PixelFormat::Gray8, data.clone()).unwrap();
        let bytes = grey
            .encode_webp(SaveOptions::default())
            .expect("a one-band raster encodes");
        let back = decode_bytes_with_limits(&bytes, DecodeLimits::default()).expect("it decodes");
        assert_eq!(back.format(), PixelFormat::Rgb8);
        let expected: Vec<Vec<u8>> = data.iter().map(|v| vec![*v, *v, *v]).collect();
        assert_eq!(pixels(&back), expected);
    }

    /**
     * Tests the `keep` contract on both ends: with metadata attached and
     * [`Keep::All`] the three blobs land in their RIFF chunks and come
     * back off the round trip byte for byte, and with [`Keep::None`] the
     * file drops to the simple `VP8L` container with no chunks at all.
     * Works by attaching all three blobs, encoding twice, and reading the
     * chunk directory of each result.
     * Input: 4x3 Rgb8 with an ICC, an EXIF and an XMP blob -> Output:
     * `VP8X`/`ICCP`/`VP8L`/`EXIF`/`XMP ` under `Keep::All` with every
     * blob recovered, and a lone `VP8L` under `Keep::None`.
     */
    #[test]
    fn keep_all_writes_the_metadata_chunks_and_keep_none_drops_them() {
        let icc: Vec<u8> = (0x10u8..=0x27).collect();
        let exif = b"II*\x00\x08\x00\x00\x00\x00\x00".to_vec();
        let xmp = b"<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"/>".to_vec();
        let mut im = ramp_rgb();
        im.fields
            .set("icc-profile-data", MetadataValue::Blob(icc.clone()));
        im.fields
            .set("exif-data", MetadataValue::Blob(exif.clone()));
        im.fields.set("xmp-data", MetadataValue::Blob(xmp.clone()));

        let kept = im
            .encode_webp(SaveOptions {
                keep: Keep::All,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(
            riff_chunks(&kept),
            vec!["VP8X", "ICCP", "VP8L", "EXIF", "XMP "],
            "the same chunk order vips webpsave writes"
        );
        let back = decode_bytes_with_limits(&kept, DecodeLimits::default()).unwrap();
        assert_eq!(back.icc_profile(), Some(&icc[..]));
        assert_eq!(
            back.get_field("exif-data"),
            Some(MetadataValue::Blob(exif.clone()))
        );
        assert_eq!(back.get_field("xmp-data"), Some(MetadataValue::Blob(xmp)));
        assert_eq!(back.data(), im.data());

        let bare = im
            .encode_webp(SaveOptions {
                keep: Keep::None,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(riff_chunks(&bare), vec!["VP8L"], "the simple container");
        let stripped = decode_bytes_with_limits(&bare, DecodeLimits::default()).unwrap();
        assert_eq!(stripped.icc_profile(), None);
        assert_eq!(stripped.get_field("exif-data"), None);
        assert_eq!(stripped.data(), im.data());
    }

    /**
     * Tests that the width and height ceiling is libwebp's 16383 rather
     * than `image-webp` 0.2.4's 16384. `vips webpsave` refuses a
     * 16384-wide image with `image too large` (`webpsave.c:740-742`) and
     * accepts 16383, so a file the crate would happily write at 16384 is
     * one the reference decoder cannot read. Works by encoding a 1-pixel
     * tall raster at each width; only the geometry matters, so the rows
     * stay cheap.
     * Input: 16383x1 and 16384x1 Rgb8 -> Output: bytes for the first, an
     * `EncodeError::Encode` naming both the geometry and the ceiling for
     * the second.
     */
    #[test]
    fn width_ceiling_is_libwebps_16383_not_the_crates_16384() {
        let ok = Raster::zeroed(MAX_DIMENSION, 1, PixelFormat::Rgb8).unwrap();
        assert!(ok.encode_webp(SaveOptions::default()).is_ok());

        let too_wide = Raster::zeroed(MAX_DIMENSION + 1, 1, PixelFormat::Rgb8).unwrap();
        let err = too_wide
            .encode_webp(SaveOptions::default())
            .expect_err("libwebp refuses 16384");
        assert!(matches!(err, EncodeError::Encode(_)), "{err:?}");
        let msg = err.to_string();
        assert!(msg.contains("16384x1"), "{msg}");
        assert!(msg.contains("16383"), "{msg}");
    }

    /**
     * Tests that the decode limits reach this decoder rather than
     * stopping at the `image` facade the sniff route used to hand WebP
     * to. Works by decoding the animation capture under a coordinate
     * ceiling and a pixel ceiling below its 4x3 geometry, and checking
     * each reports its own typed variant.
     * Input: `ANIM3` under `max_coord = 2`, under `max_pixels = 4`, and
     * under `max_alloc_bytes = 8` -> Output: `CoordLimitExceeded` and
     * `DimensionLimitExceeded` naming 4x3, and `InsufficientMemory` for
     * the frame buffer.
     */
    #[test]
    fn decode_limits_are_enforced_on_the_declared_geometry() {
        let tight = DecodeLimits::default().with_max_coord(2);
        assert!(matches!(
            decode_webp(&ANIM3, tight),
            Err(SourceError::CoordLimitExceeded {
                width: 4,
                height: 3,
                max_coord: 2
            })
        ));
        let small = DecodeLimits::default().with_max_pixels(4);
        assert!(matches!(
            decode_webp(&ANIM3, small),
            Err(SourceError::DimensionLimitExceeded {
                width: 4,
                height: 3,
                max_pixels: 4
            })
        ));
        // The allocation budget is separate: 12 pixels are inside every
        // pixel ceiling above and still need 36 bytes of frame buffer.
        //
        // It reports libviprs's own shape, with the geometry and the price it
        // computed. This used to fabricate an `image::ImageError::Limits` to
        // look like the three formats `image` refuses from inside its own
        // decoder, which threw away both (issue #686).
        let starved = DecodeLimits::default().with_max_alloc_bytes(8);
        let err = decode_webp(&ANIM3, starved).expect_err("8 bytes is not a 4x3 RGB frame");
        assert!(
            matches!(
                err,
                SourceError::AllocLimitExceeded {
                    what: "WebP frame buffer",
                    geometry: Some(DeclaredGeometry {
                        width: 4,
                        height: 3,
                        bands: 3,
                    }),
                    needed_bytes: 36,
                    max_alloc_bytes: 8,
                }
            ),
            "{err:?}"
        );
    }

    /**
     * Tests that a truncated or non-WebP buffer is refused with a typed
     * error rather than a panic or a zero-sized raster, since these bytes
     * are the untrusted end of the crate. Works by feeding the decoder a
     * prefix of a valid file, a RIFF container that is not WebP, and an
     * empty buffer.
     * Input: three malformed buffers -> Output: an `Err` from each, and
     * never a raster.
     */
    #[test]
    fn malformed_input_is_refused_with_a_typed_error() {
        for (name, bytes) in [
            ("truncated mid-chunk", &LOSSLESS_RGB[..20]),
            ("riff but not webp", &b"RIFF\x00\x00\x00\x00WAVEfmt "[..]),
            ("empty", &[][..]),
        ] {
            assert!(
                decode_webp(bytes, DecodeLimits::default()).is_err(),
                "{name} should not decode"
            );
        }
    }

    /**
     * Pins the shape of the options struct: both fields default to the
     * mode with an encoder behind it, and the struct is open enough to
     * build with `..Default::default()` from outside its own module.
     * Works by comparing `SaveOptions::default()` against an explicit
     * literal and against two functional-update literals.
     * Input: none -> Output: every spelling compares equal, with
     * compression `Lossless` and keep `All`.
     */
    #[test]
    fn save_options_default_is_lossless_keep_all_and_updatable() {
        let explicit = SaveOptions {
            compression: Compression::Lossless,
            keep: Keep::All,
        };
        let updated = SaveOptions {
            ..Default::default()
        };
        let partial = SaveOptions {
            keep: Keep::None,
            ..Default::default()
        };
        assert_eq!(SaveOptions::default(), explicit);
        assert_eq!(updated, explicit);
        assert_eq!(partial.compression, Compression::Lossless);
        assert_eq!(Compression::default(), Compression::Lossless);
        assert_eq!(Keep::default(), Keep::All);
    }

    /// The RIFF chunk tags of an encoded buffer, in file order.
    fn riff_chunks(bytes: &[u8]) -> Vec<String> {
        assert_eq!(&bytes[..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WEBP");
        let mut out = Vec::new();
        let mut p = 12;
        while p + 8 <= bytes.len() {
            out.push(String::from_utf8_lossy(&bytes[p..p + 4]).into_owned());
            let size = u32::from_le_bytes(bytes[p + 4..p + 8].try_into().unwrap()) as usize;
            p = match next_chunk(p + 8, size) {
                Some(next) => next,
                None => break,
            };
        }
        out
    }

    // -----------------------------------------------------------------
    // Animated load (issue #569). Every number below is `vips` 8.18.6 on
    // the fixture beside it; the roll it was written from is a 4x12 RGB
    // ramp with `page-height 3`, `delay 45 67 200 12` and `loop 3`.
    // -----------------------------------------------------------------

    /// `vips webpsave --lossless --keep none` on a 4x12 toilet roll
    /// carrying `page-height 3`, `delay 45 67 200 12` and `loop 3`: an
    /// animation of four 4x3 frames with four *different* durations and a
    /// finite loop count.
    ///
    /// `ANIM3` cannot do this job. Its delays are all 100 ms and its loop
    /// is 0, so a loader that read one delay and repeated it, or that lost
    /// the loop count entirely, would pass every assertion made against
    /// it. Four distinct delays also separate "the delay of loaded page i"
    /// from "the delay of file page i", which is the one place this crate
    /// diverges from vips.
    ///
    /// 45 ms is chosen for the second job it does: `rint(45 / 10)` is 4
    /// under round-half-to-even and 5 under half-up, so the `gif-delay`
    /// this file produces tells the two apart. 12 ms is chosen because it
    /// clears `webpsave`'s 10 ms floor by two.
    const ANIM4_DELAY: [u8; 488] = [
        0x52, 0x49, 0x46, 0x46, 0xe0, 0x01, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50, 0x56, 0x50, 0x38,
        0x58, 0x0a, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x02, 0x00, 0x00,
        0x41, 0x4e, 0x49, 0x4d, 0x06, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x03, 0x00, 0x41,
        0x4e, 0x4d, 0x46, 0x66, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00,
        0x00, 0x02, 0x00, 0x00, 0x2d, 0x00, 0x00, 0x02, 0x56, 0x50, 0x38, 0x4c, 0x4d, 0x00, 0x00,
        0x00, 0x2f, 0x03, 0x80, 0x00, 0x00, 0x5f, 0xa0, 0xa8, 0x6d, 0x24, 0x67, 0x7f, 0x2d, 0xfc,
        0xa1, 0x6e, 0xbd, 0x03, 0xa2, 0xa8, 0x6d, 0x24, 0x67, 0x5b, 0x18, 0x6c, 0x3b, 0xfe, 0xe0,
        0x0e, 0xc4, 0x7e, 0xd4, 0xa6, 0x81, 0x02, 0x09, 0xe9, 0x7c, 0xf6, 0x1f, 0xd4, 0xfa, 0x60,
        0xfe, 0x43, 0xa4, 0x02, 0x00, 0xf0, 0xb3, 0xe5, 0x45, 0x5e, 0xe8, 0x3b, 0x00, 0x00, 0xec,
        0x7f, 0x20, 0x0b, 0x30, 0x51, 0x18, 0xcb, 0x60, 0x52, 0x8a, 0xa5, 0x3f, 0xd8, 0x88, 0xfe,
        0x07, 0xe6, 0x1d, 0x00, 0x41, 0x4e, 0x4d, 0x46, 0x68, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x02, 0x00, 0x00, 0x43, 0x00, 0x00, 0x00, 0x56, 0x50,
        0x38, 0x4c, 0x50, 0x00, 0x00, 0x00, 0x2f, 0x03, 0x80, 0x00, 0x00, 0x5f, 0xa0, 0xa6, 0x8d,
        0x24, 0xa7, 0xbc, 0x30, 0xf5, 0xf1, 0x47, 0x99, 0xff, 0x81, 0xa8, 0x85, 0x24, 0x09, 0x8a,
        0xc6, 0x60, 0xf7, 0x1e, 0xa6, 0xf3, 0x67, 0x38, 0x88, 0x4d, 0x94, 0xb4, 0x91, 0x04, 0x81,
        0xb2, 0x7b, 0x74, 0x9c, 0x7f, 0x6d, 0x87, 0x37, 0x99, 0xff, 0x88, 0xbb, 0x00, 0x00, 0x7c,
        0x95, 0x4c, 0x64, 0x42, 0x76, 0x01, 0x00, 0xb0, 0xff, 0x40, 0x16, 0x60, 0xa2, 0x30, 0x96,
        0xc1, 0xa4, 0x14, 0x4b, 0x7f, 0xb0, 0x11, 0xfd, 0x0f, 0xcc, 0x3b, 0x41, 0x4e, 0x4d, 0x46,
        0x6a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x02, 0x00,
        0x00, 0xc8, 0x00, 0x00, 0x00, 0x56, 0x50, 0x38, 0x4c, 0x51, 0x00, 0x00, 0x00, 0x2f, 0x03,
        0x80, 0x00, 0x00, 0x5f, 0xa0, 0xa6, 0x6d, 0x24, 0xe6, 0xdb, 0x96, 0xfd, 0xb1, 0x3c, 0x7f,
        0x4e, 0x95, 0x89, 0x9a, 0x36, 0x92, 0x9c, 0xea, 0x22, 0x7f, 0x50, 0xc7, 0x24, 0xfd, 0x33,
        0x98, 0x46, 0x6d, 0xdb, 0x36, 0xcc, 0x60, 0x3b, 0xb7, 0xe6, 0xff, 0xb6, 0x09, 0x98, 0xff,
        0xa8, 0xea, 0x88, 0x20, 0x82, 0xc8, 0xf7, 0x68, 0xd0, 0xa0, 0x41, 0xfc, 0xd7, 0x20, 0xb3,
        0x20, 0x02, 0x82, 0x00, 0x12, 0x85, 0x58, 0x72, 0x98, 0x65, 0x96, 0x95, 0x26, 0x8d, 0xe8,
        0x7f, 0xec, 0x0d, 0x02, 0x00, 0x41, 0x4e, 0x4d, 0x46, 0x64, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x02, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00, 0x56,
        0x50, 0x38, 0x4c, 0x4c, 0x00, 0x00, 0x00, 0x2f, 0x03, 0x80, 0x00, 0x00, 0x5f, 0xa0, 0x26,
        0x00, 0x01, 0xe6, 0x35, 0x7a, 0xe9, 0x9f, 0xc3, 0xef, 0x13, 0x44, 0x4d, 0xdb, 0x46, 0xd0,
        0x64, 0x06, 0xc7, 0x1f, 0xd7, 0x21, 0xf9, 0x17, 0x35, 0x6d, 0x1b, 0x41, 0x83, 0xb7, 0xc2,
        0x3e, 0xbc, 0x87, 0xe0, 0x93, 0xe6, 0x3f, 0x72, 0xd5, 0x30, 0x8c, 0x47, 0x18, 0xad, 0xb1,
        0x7f, 0xc8, 0x66, 0xb4, 0xc6, 0xfe, 0x81, 0x20, 0x80, 0x44, 0x21, 0x96, 0x58, 0x76, 0x58,
        0x69, 0x87, 0x85, 0x23, 0xfa, 0x1f, 0x1f, 0x36,
    ];

    /// The 144 bytes `vips rawsave 'roll4d.webp[n=-1]'` writes: the whole
    /// four-page roll, top page first, which is the layout this crate calls
    /// a [`PageLayout`](crate::frames::PageLayout).
    const ANIM4_ROLL: [u8; 144] = [
        0, 0, 0, 5, 11, 3, 10, 22, 6, 15, 33, 9, 25, 7, 13, 30, 18, 16, 35, 29, 19, 40, 40, 22, 50,
        14, 26, 55, 25, 29, 60, 36, 32, 65, 47, 35, 75, 21, 39, 80, 32, 42, 85, 43, 45, 90, 54, 48,
        100, 28, 52, 105, 39, 55, 110, 50, 58, 115, 61, 61, 125, 35, 65, 130, 46, 68, 135, 57, 71,
        140, 68, 74, 150, 42, 78, 155, 53, 81, 160, 64, 84, 165, 75, 87, 175, 49, 91, 180, 60, 94,
        185, 71, 97, 190, 82, 100, 200, 56, 104, 205, 67, 107, 210, 78, 110, 215, 89, 113, 225, 63,
        117, 230, 74, 120, 235, 85, 123, 240, 96, 126, 250, 70, 130, 255, 81, 133, 4, 92, 136, 9,
        103, 139, 19, 77, 143, 24, 88, 146, 29, 99, 149, 34, 110, 152,
    ];

    /// Every frame of an animation, which is `n = -1` in vips.
    fn all_pages() -> LoadOptions {
        LoadOptions::default().with_n(-1)
    }

    /**
     * Tests that asking for every frame stacks them into one toilet-roll
     * raster with the page geometry the frames model derives, rather than
     * handing back frame 0 the way the still lane did. Works by decoding
     * the four-frame capture with `n = -1` and comparing the whole
     * buffer to what `vips rawsave 'x.webp[n=-1]'` wrote.
     * Input: `ANIM4_DELAY` with `n = -1` -> Output: a 4x12 `Rgb8` raster
     * whose bytes are `ANIM4_ROLL`, four pages of three rows, with
     * `n-pages` 4 as vips reports.
     */
    #[test]
    fn every_frame_stacks_into_one_roll() {
        let raster = decode_webp_with(&ANIM4_DELAY, DecodeLimits::default(), all_pages())
            .expect("the four-frame capture decodes");
        assert_eq!((raster.width(), raster.height()), (4, 12));
        assert_eq!(raster.format(), PixelFormat::Rgb8);
        assert_eq!(raster.data(), &ANIM4_ROLL[..]);
        // The split the loader wrote and the split the reader derives are
        // the same one: `vipsheader -a 'x.webp[n=-1]'` reports
        // `page-height: 3` and `n-pages: 4`.
        assert_eq!(raster.get_page_height(), 3);
        assert_eq!(raster.pages_loaded(), 4);
        assert_eq!(raster.get_n_pages(), 4);
    }

    /**
     * Tests that the per-frame delays land as milliseconds with no
     * conversion, which is what separates this loader from the GIF one,
     * and that the loop count arrives unshifted. Works by reading the
     * `delay`, `loop`, `gif-delay` and `gif-loop` fields off a whole-file
     * load and comparing them to `vipsheader -f` on the same bytes.
     * Input: `ANIM4_DELAY` with `n = -1` -> Output: `delay` = `[45, 67,
     * 200, 12]`, `loop` = 3, `gif-delay` = 4, `gif-loop` = 2.
     */
    #[test]
    fn the_delays_are_milliseconds_and_the_loop_count_is_unshifted() {
        let raster = decode_webp_with(&ANIM4_DELAY, DecodeLimits::default(), all_pages())
            .expect("the four-frame capture decodes");
        // `vipsheader -f delay` prints `45 67 200 12 `. The WebP `ANMF`
        // duration is milliseconds on the wire, so nothing is divided by
        // ten on the way in; the GIF loader is the one that has to.
        assert_eq!(
            raster.get_int_array("delay"),
            Some(&[45i64, 67, 200, 12][..])
        );
        // `vipsheader -f loop` prints 3, and the `ANIM` chunk holds 3 too:
        // WebP counts plays with no off-by-one, where GIF's NETSCAPE block
        // counts repeats.
        assert_eq!(raster.get_field("loop"), Some(MetadataValue::Int(3)));
        // The two compatibility fields vips attaches beside them. 45 ms is
        // 4.5 centiseconds and `gif-delay` is 4, not 5, so the rounding is
        // half-to-even, the same rule `FrameDelay::to_centiseconds` was
        // measured into.
        assert_eq!(raster.get_field("gif-delay"), Some(MetadataValue::Int(4)));
        // `gif-loop` counts repeats after the first play, so 3 plays is 2.
        assert_eq!(raster.get_field("gif-loop"), Some(MetadataValue::Int(2)));
    }

    /**
     * Tests that a partial load subsets the delay array to the pages it
     * actually loaded, which is the one place this loader deliberately
     * disagrees with vips. Works by loading two frames from the middle of
     * a four-frame file and asserting the array is those two frames'
     * delays rather than the file's four.
     * Input: `ANIM4_DELAY` with `page = 1, n = Some(2)` -> Output: a 4x6
     * two-page raster whose `delay` is `[67, 200]`, where vips reports the
     * file's whole `45 67 200 12`.
     */
    #[test]
    fn a_partial_load_subsets_the_delay_array() {
        let raster = decode_webp_with(
            &ANIM4_DELAY,
            DecodeLimits::default(),
            LoadOptions::default().with_page(1).with_n(2),
        )
        .expect("frames 1 and 2 exist");
        assert_eq!((raster.width(), raster.height()), (4, 6));
        assert_eq!(raster.data(), &ANIM4_ROLL[36..108]);
        assert_eq!(raster.pages_loaded(), 2);
        assert_eq!(raster.get_page_height(), 3);
        // Measured: `vipsheader -f delay 'x.webp[page=1,n=2]'` prints
        // `45 67 200 12 ` for this exact load, which are the delays of
        // pages 0..4 attached to a raster holding pages 1 and 2. Nothing
        // on the raster records the offset, so a saver reading that array
        // writes 45 and 67 onto frames that are really 1 and 2. Making
        // `delay[i]` the delay of *loaded* page `i` is what makes the
        // array usable at all.
        assert_eq!(raster.get_int_array("delay"), Some(&[67i64, 200][..]));
        // `gif-delay` is the first delay in centiseconds, so it follows
        // the subset too: 67 ms is 7 cs, where vips reports 4.
        assert_eq!(raster.get_field("gif-delay"), Some(MetadataValue::Int(7)));
        // `n-pages` does *not* follow the subset. It is the file's count
        // and vips reports 4 here as well (issue #635).
        assert_eq!(raster.get_n_pages(), 4);
    }

    /**
     * Tests that the delay array always has exactly one entry per loaded
     * page, which is the invariant that makes it usable and the thing
     * vips's file-scoped array does not have. Works by sweeping every
     * page/count combination a four-frame file accepts and comparing the
     * array length to `pages_loaded()`.
     * Input: `ANIM4_DELAY` over ten accepted requests -> Output: `delay`
     * as long as the raster's own page count, every time.
     */
    #[test]
    fn the_delay_array_is_always_as_long_as_the_pages_loaded() {
        let requests: [(u32, i32); 10] = [
            (0, 1),
            (0, 2),
            (0, 4),
            (0, -1),
            (1, 1),
            (1, 3),
            (1, -1),
            (2, 2),
            (3, 1),
            (3, -1),
        ];
        for (page, n) in requests {
            let raster = decode_webp_with(
                &ANIM4_DELAY,
                DecodeLimits::default(),
                LoadOptions::default().with_page(page).with_n(n),
            )
            .unwrap_or_else(|e| panic!("page={page} n={n} is in range: {e}"));
            let delays = raster
                .get_int_array("delay")
                .unwrap_or_else(|| panic!("page={page} n={n} must carry a delay"));
            assert_eq!(
                delays.len() as u32,
                raster.pages_loaded(),
                "page={page} n={n}: one delay per loaded page"
            );
            let expected: Vec<i64> =
                [45i64, 67, 200, 12][page as usize..page as usize + delays.len()].to_vec();
            assert_eq!(delays, expected, "page={page} n={n}");
        }
    }

    /**
     * Tests that a one-page load carries no page split at all, which is
     * what vips does and what stops a single frame reading as an
     * animation. Works by loading each page on its own and asserting the
     * geometry, the missing field and the one-entry delay.
     * Input: `ANIM4_DELAY` at each `page` with `n = Some(1)` -> Output:
     * 4x3, no `page-height` field, `pages_loaded` 1, and that page's own
     * delay.
     */
    #[test]
    fn a_one_page_load_carries_no_page_split() {
        for (page, delay) in [(0u32, 45i64), (1, 67), (2, 200), (3, 12)] {
            let raster = decode_webp_with(
                &ANIM4_DELAY,
                DecodeLimits::default(),
                LoadOptions::default().with_page(page),
            )
            .expect("every page of a four-frame file loads");
            assert_eq!((raster.width(), raster.height()), (4, 3), "page {page}");
            // Measured: `vipsheader -f page-height 'x.webp[page=1]'` fails
            // with `field "page-height" not found`, and so does a default
            // load. The field appears only when more than one page is in
            // the raster.
            assert_eq!(raster.get_field("page-height"), None, "page {page}");
            assert_eq!(raster.pages_loaded(), 1, "page {page}");
            assert_eq!(
                raster.data(),
                &ANIM4_ROLL[page as usize * 36..page as usize * 36 + 36],
                "page {page}"
            );
            assert_eq!(
                raster.get_int_array("delay"),
                Some(&[delay][..]),
                "page {page}"
            );
        }
    }

    /**
     * Tests that the default load is still exactly what it was, so the
     * animation work did not move the still path underneath anyone. Works
     * by decoding the four-frame capture through the two-argument entry
     * point and comparing to page 0.
     * Input: `ANIM4_DELAY` through `decode_webp` -> Output: 4x3, frame 0's
     * pixels, `n-pages` 4, and the new `delay` field holding one entry.
     */
    #[test]
    fn the_default_load_is_still_frame_zero() {
        let raster = decode_webp(&ANIM4_DELAY, DecodeLimits::default())
            .expect("the default load reads one frame");
        assert_eq!((raster.width(), raster.height()), (4, 3));
        assert_eq!(raster.data(), &ANIM4_ROLL[..36]);
        assert_eq!(raster.get_n_pages(), 4);
        assert_eq!(raster.get_field("page-height"), None);
        assert_eq!(raster.get_int_array("delay"), Some(&[45i64][..]));
        assert_eq!(
            LoadOptions::default(),
            LoadOptions::default().with_page(0).with_n(1)
        );
    }

    /**
     * Tests that a page past the end of the file is refused with the file's
     * own count rather than clamped, matching what vips does and not what
     * a forgiving loader would do. Works by asking for the last page, then
     * one past it, then a count that runs off the end, then no pages at
     * all.
     * Input: `ANIM4_DELAY` at `page = 3`, `page = 4`, `page = 2, n = 5`
     * and `n = 0` -> Output: the last page loads; the other three are
     * `SourceError::PageOutOfRange` naming four pages.
     */
    #[test]
    fn a_page_past_the_end_is_refused_rather_than_clamped() {
        // `vips copy 'x.webp[page=3]'` gives 4x3: three is the last index
        // of a four-page file.
        assert!(
            decode_webp_with(
                &ANIM4_DELAY,
                DecodeLimits::default(),
                LoadOptions::default().with_page(3).with_n(1),
            )
            .is_ok()
        );
        // And `[page=4]`, `[page=2,n=5]` and `[n=0]` all fail with
        // `webp: bad page number`.
        for (page, n) in [(4u32, 1i32), (2, 5), (0, 0), (9, -1), (0, -2)] {
            let err = decode_webp_with(
                &ANIM4_DELAY,
                DecodeLimits::default(),
                LoadOptions::default().with_page(page).with_n(n),
            )
            .expect_err("vips calls this a bad page number");
            assert!(
                matches!(
                    err,
                    SourceError::PageOutOfRange {
                        format: "webp",
                        pages: 4,
                        ..
                    }
                ),
                "page={page} n={n} got {err:?}"
            );
        }
    }

    /**
     * Tests that a still image has no animation surface at all: no delay,
     * no loop, and no page but its own. Works by loading the lossless
     * still capture through every option shape and asserting the fields
     * are absent and page 1 is refused.
     * Input: `LOSSLESS_RGB` -> Output: 4x3 with none of `delay`, `loop`,
     * `gif-delay`, `gif-loop` or `n-pages`, and a refusal for page 1.
     */
    #[test]
    fn a_still_image_has_no_animation_fields() {
        // `vipsheader -a still.webp` lists none of these, and neither does
        // `still.webp[n=-1]`.
        for options in [LoadOptions::default(), all_pages()] {
            let raster = decode_webp_with(&LOSSLESS_RGB, DecodeLimits::default(), options)
                .expect("a still loads under either option shape");
            assert_eq!((raster.width(), raster.height()), (4, 3));
            for field in ["delay", "loop", "gif-delay", "gif-loop", "n-pages"] {
                assert_eq!(raster.get_field(field), None, "{field} on a still");
            }
        }
        let err = decode_webp_with(
            &LOSSLESS_RGB,
            DecodeLimits::default(),
            LoadOptions::default().with_page(1).with_n(1),
        )
        .expect_err("a still has one page");
        assert!(
            matches!(err, SourceError::PageOutOfRange { pages: 1, .. }),
            "got {err:?}"
        );
    }

    /**
     * Tests that an animation saved without any delay comes back with the
     * hundred-millisecond floor `webpsave` writes and a forever loop, so
     * the clamp is pinned as a property of the wire rather than of this
     * loader. Works by loading the three-frame default-save capture and
     * reading the fields.
     * Input: `ANIM3` with `n = -1` -> Output: 4x9, `delay` = `[100, 100,
     * 100]`, `loop` = 0, `gif-delay` = 10, `gif-loop` = 0.
     */
    #[test]
    fn a_default_save_carries_the_browser_floor_and_a_forever_loop() {
        let raster = decode_webp_with(&ANIM3, DecodeLimits::default(), all_pages())
            .expect("the three-frame capture decodes");
        assert_eq!((raster.width(), raster.height()), (4, 9));
        assert_eq!(raster.pages_loaded(), 3);
        // The roll was saved with no `delay` attached, and `webpsave`
        // wrote 100 ms into every `ANMF`: the floor is applied on save, so
        // the file itself holds the hundred and this loader reads it back
        // rather than inventing it.
        assert_eq!(raster.get_int_array("delay"), Some(&[100i64, 100, 100][..]));
        // `loop 0` is play-forever, and the `ANIM` chunk holds 0 for it.
        assert_eq!(raster.get_field("loop"), Some(MetadataValue::Int(0)));
        assert_eq!(raster.get_field("gif-delay"), Some(MetadataValue::Int(10)));
        // Forever is 0 repeats as well as 0 plays, so `gif-loop` is 0 for
        // both `loop 0` and `loop 1`; only the play count separates them.
        assert_eq!(raster.get_field("gif-loop"), Some(MetadataValue::Int(0)));
    }

    /**
     * Tests that the page split the loader writes is one the page model
     * can actually read back, so a frame extracted from the roll is the
     * frame a single-page load hands over. Works by extracting each page
     * of a whole-file load and comparing it to a load of that page alone.
     * Input: `ANIM4_DELAY` -> Output: `try_extract_page(i)` equals the
     * `page = i` load, byte for byte, for all four pages.
     */
    #[test]
    fn a_page_of_the_roll_is_the_page_loaded_on_its_own() {
        let roll = decode_webp_with(&ANIM4_DELAY, DecodeLimits::default(), all_pages())
            .expect("the four-frame capture decodes");
        // Asserted before the loop, because a roll holding one page makes
        // the comparison below trivially true.
        assert_eq!(roll.pages_loaded(), 4);
        for page in 0..roll.pages_loaded() {
            let extracted = roll.try_extract_page(page).expect("page is in range");
            let alone = decode_webp_with(
                &ANIM4_DELAY,
                DecodeLimits::default(),
                LoadOptions::default().with_page(page),
            )
            .expect("page is in range");
            assert_eq!(
                (extracted.width(), extracted.height()),
                (alone.width(), alone.height()),
                "page {page}"
            );
            assert_eq!(extracted.data(), alone.data(), "page {page}");
        }
    }

    /**
     * Tests that the resource ceilings are checked against the whole roll
     * and not against one frame, which is the difference between a
     * four-frame load and the single-frame load they were written for.
     * Works by setting each ceiling so one frame fits and four do not, and
     * asserting the refusal names the roll's geometry.
     * Input: `ANIM4_DELAY` under `max_pixels = 40`, `max_coord = 6` and a
     * 100-byte allocation budget -> Output: a typed refusal in each case,
     * with a positive control at `n = Some(1)` that still loads.
     */
    #[test]
    fn the_ceilings_are_checked_against_the_roll_not_the_frame() {
        // 4x3 is 12 pixels and 4x12 is 48, so a 40-pixel ceiling separates
        // one frame from four.
        let pixels = DecodeLimits::default().with_max_pixels(40);
        assert!(
            decode_webp_with(&ANIM4_DELAY, pixels, LoadOptions::default()).is_ok(),
            "one frame fits under 40 pixels"
        );
        assert!(
            matches!(
                decode_webp_with(&ANIM4_DELAY, pixels, all_pages()),
                Err(SourceError::DimensionLimitExceeded { height: 12, .. })
            ),
            "four frames do not"
        );

        // The single-axis ceiling sees the roll's 12 rows, not the frame's
        // 3.
        let coord = DecodeLimits::default().with_max_coord(6);
        assert!(decode_webp_with(&ANIM4_DELAY, coord, LoadOptions::default()).is_ok());
        assert!(matches!(
            decode_webp_with(&ANIM4_DELAY, coord, all_pages()),
            Err(SourceError::CoordLimitExceeded { height: 12, .. })
        ));

        // And the allocation budget prices four frames, 144 bytes, not one
        // frame's 36.
        let alloc = DecodeLimits::default().with_max_alloc_bytes(100);
        assert!(decode_webp_with(&ANIM4_DELAY, alloc, LoadOptions::default()).is_ok());
        let err = decode_webp_with(&ANIM4_DELAY, alloc, all_pages())
            .expect_err("144 bytes is over a 100-byte budget");
        assert!(
            matches!(
                err,
                SourceError::AllocLimitExceeded {
                    needed_bytes: 144,
                    geometry: Some(DeclaredGeometry {
                        width: 4,
                        height: 12,
                        bands: 3
                    }),
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    /// `vips webpsave --lossless --keep none` on the same 4x12 roll with a
    /// fourth, independent alpha channel: four 4x3 `VP8L` frames that
    /// declare `alpha_is_used` and that vips wrote with blending switched
    /// **off** on every one of them.
    ///
    /// The negative control for [`disable_blending_on_opaque_frames`]. The
    /// opaque capture has blending on for frames 1, 2 and 3 and this one
    /// has it off everywhere, so the rewrite has three frames to touch
    /// there and none here, and the pixels have to be exact either way.
    const ANIM4_RGBA: [u8; 550] = [
        0x52, 0x49, 0x46, 0x46, 0x1e, 0x02, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50, 0x56, 0x50, 0x38,
        0x58, 0x0a, 0x00, 0x00, 0x00, 0x12, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x02, 0x00, 0x00,
        0x41, 0x4e, 0x49, 0x4d, 0x06, 0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0x03, 0x00, 0x41,
        0x4e, 0x4d, 0x46, 0x74, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00,
        0x00, 0x02, 0x00, 0x00, 0x2d, 0x00, 0x00, 0x02, 0x56, 0x50, 0x38, 0x4c, 0x5b, 0x00, 0x00,
        0x00, 0x2f, 0x03, 0x80, 0x00, 0x10, 0x5f, 0xa0, 0xa8, 0x6d, 0x24, 0x67, 0x7f, 0x2d, 0xfc,
        0xa1, 0x6e, 0xbd, 0x03, 0xa2, 0xa8, 0x6d, 0x24, 0x67, 0x5b, 0x18, 0x6c, 0x3b, 0xfe, 0xe0,
        0x0e, 0xc4, 0x7e, 0xd4, 0xa6, 0x81, 0x02, 0x09, 0xe9, 0x7c, 0xf6, 0x1f, 0xd4, 0xfa, 0x40,
        0x41, 0xdb, 0x48, 0xc8, 0x83, 0x82, 0x03, 0xff, 0x0a, 0x1f, 0x2c, 0x84, 0x48, 0x05, 0x00,
        0x00, 0xc0, 0xcf, 0xee, 0xbc, 0x7c, 0x5e, 0x5e, 0xdf, 0x19, 0x00, 0x00, 0x60, 0xff, 0x0f,
        0xc8, 0x02, 0x4c, 0x14, 0xc6, 0x32, 0x98, 0x94, 0x62, 0xe9, 0x0f, 0x36, 0xa2, 0xff, 0x81,
        0x79, 0x07, 0x00, 0x41, 0x4e, 0x4d, 0x46, 0x78, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x03, 0x00, 0x00, 0x02, 0x00, 0x00, 0x43, 0x00, 0x00, 0x02, 0x56, 0x50, 0x38,
        0x4c, 0x60, 0x00, 0x00, 0x00, 0x2f, 0x03, 0x80, 0x00, 0x10, 0x5f, 0xa0, 0xa6, 0x8d, 0x24,
        0xa7, 0xbc, 0x30, 0xf5, 0xf1, 0x47, 0x99, 0xff, 0x81, 0xa8, 0x85, 0x24, 0x09, 0x8a, 0xc6,
        0x60, 0xf7, 0x1e, 0xa6, 0xf3, 0x67, 0x38, 0x88, 0x4d, 0x94, 0xb4, 0x91, 0x04, 0x81, 0xb2,
        0x7b, 0x74, 0x9c, 0x7f, 0x6d, 0x87, 0x37, 0x51, 0xc8, 0x48, 0x12, 0xf3, 0x85, 0xe0, 0xee,
        0x1e, 0xe7, 0xfd, 0x69, 0x9a, 0x42, 0xc4, 0x5d, 0x03, 0x00, 0x00, 0x7c, 0x55, 0x66, 0xf2,
        0x99, 0xbc, 0xec, 0x16, 0x00, 0x00, 0xc0, 0xfe, 0x0f, 0xc8, 0x02, 0x4c, 0x14, 0xc6, 0x32,
        0x98, 0x94, 0x62, 0xe9, 0x0f, 0x36, 0xa2, 0xff, 0x81, 0x79, 0x07, 0x41, 0x4e, 0x4d, 0x46,
        0x78, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x02, 0x00,
        0x00, 0xc8, 0x00, 0x00, 0x02, 0x56, 0x50, 0x38, 0x4c, 0x5f, 0x00, 0x00, 0x00, 0x2f, 0x03,
        0x80, 0x00, 0x10, 0x5f, 0xa0, 0xa6, 0x6d, 0x24, 0xe6, 0xdb, 0x96, 0xfd, 0xb1, 0x3c, 0x7f,
        0x4e, 0x95, 0x89, 0x9a, 0x36, 0x92, 0x9c, 0xea, 0x22, 0x7f, 0x50, 0xc7, 0x24, 0xfd, 0x33,
        0x98, 0x46, 0x6d, 0xdb, 0x36, 0xcc, 0x60, 0x3b, 0xb7, 0xe6, 0xff, 0xb6, 0x09, 0x50, 0xc8,
        0x46, 0x12, 0xc4, 0x30, 0x0a, 0x4b, 0xbe, 0x80, 0xcf, 0x11, 0x5c, 0x12, 0x55, 0x6d, 0x44,
        0x20, 0x02, 0x11, 0xdf, 0x93, 0x86, 0x6b, 0xb8, 0x86, 0x8b, 0xff, 0xdf, 0x70, 0x99, 0x25,
        0x44, 0x80, 0x20, 0x80, 0x44, 0x21, 0x96, 0x1c, 0x66, 0x99, 0x65, 0xa5, 0x49, 0x23, 0xfa,
        0x1f, 0x7b, 0x83, 0x00, 0x41, 0x4e, 0x4d, 0x46, 0x76, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x02, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x02, 0x56, 0x50,
        0x38, 0x4c, 0x5d, 0x00, 0x00, 0x00, 0x2f, 0x03, 0x80, 0x00, 0x10, 0x5f, 0xa0, 0x98, 0x91,
        0x24, 0x08, 0xa0, 0x29, 0x46, 0x66, 0xfd, 0x59, 0xe6, 0xf8, 0x2d, 0x88, 0x92, 0x36, 0x92,
        0x20, 0x34, 0xf8, 0x5e, 0xff, 0x92, 0x4e, 0xd0, 0x0d, 0x51, 0xd4, 0x46, 0x0a, 0xc4, 0xf1,
        0x02, 0xd9, 0x88, 0xc5, 0x45, 0x13, 0x53, 0x91, 0x6c, 0x3c, 0x24, 0xb8, 0x2c, 0xfa, 0xff,
        0x8b, 0xa0, 0xc5, 0x55, 0x88, 0xfb, 0x3b, 0x68, 0xa0, 0x81, 0xe6, 0x77, 0x67, 0x72, 0x52,
        0x31, 0xeb, 0xfb, 0x9f, 0x9c, 0x9c, 0x04, 0x41, 0x00, 0x89, 0x42, 0x2c, 0xc1, 0x4c, 0xb2,
        0xc4, 0x2c, 0x0b, 0x47, 0xf4, 0x3f, 0x3e, 0x4c, 0x01, 0x00,
    ];

    /// The 192 bytes `vips rawsave 'rgba_roll.webp[n=-1]'` writes.
    const ANIM4_RGBA_ROLL: [u8; 192] = [
        0, 0, 0, 0, 5, 11, 3, 17, 10, 22, 6, 34, 15, 33, 9, 51, 25, 7, 13, 20, 30, 18, 16, 37, 35,
        29, 19, 54, 40, 40, 22, 71, 50, 14, 26, 40, 55, 25, 29, 57, 60, 36, 32, 74, 65, 47, 35, 91,
        75, 21, 39, 60, 80, 32, 42, 77, 85, 43, 45, 94, 90, 54, 48, 111, 100, 28, 52, 80, 105, 39,
        55, 97, 110, 50, 58, 114, 115, 61, 61, 131, 125, 35, 65, 100, 130, 46, 68, 117, 135, 57,
        71, 134, 140, 68, 74, 151, 150, 42, 78, 120, 155, 53, 81, 137, 160, 64, 84, 154, 165, 75,
        87, 171, 175, 49, 91, 140, 180, 60, 94, 157, 185, 71, 97, 174, 190, 82, 100, 191, 200, 56,
        104, 160, 205, 67, 107, 177, 210, 78, 110, 194, 215, 89, 113, 211, 225, 63, 117, 180, 230,
        74, 120, 197, 235, 85, 123, 214, 240, 96, 126, 231, 250, 70, 130, 200, 255, 81, 133, 217,
        4, 92, 136, 234, 9, 103, 139, 251, 19, 77, 143, 220, 24, 88, 146, 237, 29, 99, 149, 254,
        34, 110, 152, 15,
    ];

    /**
     * Tests that the blend-flag rewrite finds exactly the frames whose
     * pixels the decoder would otherwise lose a grey level on, and touches
     * nothing else. Works by running the scanner over the opaque capture,
     * the transparent one and a still, and checking the byte it names.
     * Input: `ANIM4_DELAY`, `ANIM4_RGBA` and `LOSSLESS_RGB` -> Output:
     * three offsets on the opaque animation and none on the other two, and
     * a rewrite that borrows rather than clones where there is nothing to
     * do.
     */
    #[test]
    fn the_blend_rewrite_finds_the_opaque_blended_frames_and_only_those() {
        // `vips webpsave` wrote frame 0 with blending off (frame-info 0x02)
        // and frames 1, 2 and 3 with it on (0x00), all four `VP8L` with
        // `alpha_is_used` clear. Those three are the ones the decoder gets
        // wrong.
        let offsets = opaque_blended_frame_offsets(&ANIM4_DELAY);
        assert_eq!(offsets.len(), 3, "three blended opaque frames");
        for offset in &offsets {
            assert_eq!(
                ANIM4_DELAY[*offset] & ANMF_NO_BLEND,
                0,
                "the scanner only names frames that ask to be blended"
            );
        }
        let rewritten = disable_blending_on_opaque_frames(&ANIM4_DELAY);
        assert!(matches!(rewritten, Cow::Owned(_)), "it had work to do");
        for offset in &offsets {
            assert_eq!(rewritten[*offset] & ANMF_NO_BLEND, ANMF_NO_BLEND);
        }
        // And nothing else moved: exactly three bytes differ.
        let moved = rewritten
            .iter()
            .zip(ANIM4_DELAY.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(moved, 3);

        // The transparent animation declares `alpha_is_used` on every
        // frame, and vips wrote blending off on all four anyway, so there
        // is nothing to name and nothing to clone.
        assert!(opaque_blended_frame_offsets(&ANIM4_RGBA).is_empty());
        assert!(matches!(
            disable_blending_on_opaque_frames(&ANIM4_RGBA),
            Cow::Borrowed(_)
        ));
        // A still has no `ANMF` chunk at all.
        assert!(opaque_blended_frame_offsets(&LOSSLESS_RGB).is_empty());
        assert!(matches!(
            disable_blending_on_opaque_frames(&LOSSLESS_RGB),
            Cow::Borrowed(_)
        ));
        // And a buffer that is not a RIFF container at all walks nowhere
        // rather than reading past its end.
        assert!(opaque_blended_frame_offsets(b"not a webp").is_empty());
        assert!(opaque_blended_frame_offsets(&[]).is_empty());
    }

    /**
     * Tests that a transparent animation loads byte-exact without the
     * rewrite touching it, which is the positive control the opaque case
     * needs: the pixels have to be right whether or not any flag moved.
     * Works by loading every page of the RGBA capture and comparing to
     * what `vips rawsave 'x.webp[n=-1]'` wrote.
     * Input: `ANIM4_RGBA` with `n = -1` -> Output: 4x12 `Rgba8` equal to
     * `ANIM4_RGBA_ROLL`, with the same delays and loop as the opaque roll
     * it was written from.
     */
    #[test]
    fn a_transparent_animation_loads_exactly() {
        let raster = decode_webp_with(&ANIM4_RGBA, DecodeLimits::default(), all_pages())
            .expect("the transparent capture decodes");
        assert_eq!((raster.width(), raster.height()), (4, 12));
        assert_eq!(raster.format(), PixelFormat::Rgba8);
        assert_eq!(raster.data(), &ANIM4_RGBA_ROLL[..]);
        assert_eq!(raster.pages_loaded(), 4);
        assert_eq!(raster.get_page_height(), 3);
        assert_eq!(
            raster.get_int_array("delay"),
            Some(&[45i64, 67, 200, 12][..])
        );
        assert_eq!(raster.get_field("loop"), Some(MetadataValue::Int(3)));
    }

    /**
     * Tests that a scanner asked about a frame that declares alpha and
     * asks to be blended leaves it alone, which the two captures cannot
     * show on their own: vips writes blending off on every frame of the
     * transparent roll and on with no alpha on the opaque one, so the two
     * conditions never come apart in a file the oracle produces. Works by
     * clearing the no-blend bit on the transparent capture by hand, which
     * is the combination no encoder here writes, and asking again.
     * Input: `ANIM4_RGBA` with every frame's no-blend bit cleared ->
     * Output: still no offsets, where the same edit to the opaque capture
     * names all four frames.
     */
    #[test]
    fn a_frame_that_declares_alpha_is_left_blended() {
        // Clearing the bit everywhere is the only way to reach the fourth
        // combination: `vips webpsave` writes blend-off + alpha for a
        // transparent roll and blend-on + no-alpha for an opaque one, so
        // blend-on + alpha has to be built.
        let clear_all = |bytes: &[u8]| {
            let mut owned = bytes.to_vec();
            let mut cursor = 12usize;
            while let Some(header) = owned.get(cursor..cursor + 8) {
                let size = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
                if &header[..4] == b"ANMF" {
                    owned[cursor + 8 + 15] &= !ANMF_NO_BLEND;
                }
                cursor = match next_chunk(cursor + 8, size) {
                    Some(next) => next,
                    None => break,
                };
            }
            owned
        };

        let opaque = clear_all(&ANIM4_DELAY);
        assert_eq!(
            opaque_blended_frame_offsets(&opaque).len(),
            4,
            "with frame 0 blended too, all four opaque frames need the rewrite"
        );

        let transparent = clear_all(&ANIM4_RGBA);
        assert!(
            opaque_blended_frame_offsets(&transparent).is_empty(),
            "a frame that declares alpha keeps its blending: switching it off \
             would change the image, where switching it off on an opaque frame \
             cannot"
        );
        // The edit really did happen, or the assertion above would pass on
        // a buffer nothing had touched.
        assert_ne!(transparent, ANIM4_RGBA.to_vec());
        assert_ne!(opaque, ANIM4_DELAY.to_vec());
    }

    /**
     * Tests that the RIFF walk stops on a chunk size the file's own
     * arithmetic cannot represent, rather than panicking on the pad or
     * wrapping to an offset it has already passed. Works by calling the
     * step directly at the sizes a four-byte size field can hold, because
     * the failing case cannot be reached through the walk on a 64-bit host:
     * there the same file gives an offset far past the buffer and the next
     * `get` ends the loop.
     * Input: the padded and unpadded shapes, then `usize::MAX` and the two
     * ways the outer addition can overflow -> Output: the next offset, and
     * `None` for every request that cannot be represented.
     */
    #[test]
    fn the_chunk_walk_stops_on_a_size_that_cannot_be_represented() {
        // The ordinary shapes. A `VP8L` payload of 116 bytes needs no pad
        // and one of 117 needs a byte, which is the whole of the rule.
        assert_eq!(next_chunk(20, 116), Some(136));
        assert_eq!(next_chunk(20, 117), Some(138));
        assert_eq!(next_chunk(0, 0), Some(0));

        // The pad overflows before the outer addition sees anything. This
        // is the case a hostile file supplies: `size` is read as a `u32`,
        // and on a 32-bit target `u32::MAX as usize` *is* `usize::MAX`.
        assert_eq!(next_chunk(12, usize::MAX), None);
        // Which is why that input is a real file there and not here. The
        // assertion is written as an equality rather than cfg'd away so it
        // says something true on both targets, and says which one is which.
        assert_eq!(
            next_chunk(12, u32::MAX as usize).is_none(),
            usize::BITS == 32,
            "a chunk declaring u32::MAX overflows the pad on a 32-bit target \
             and is merely past the end of the buffer on a 64-bit one"
        );

        // And the outer addition, which was already checked: an even size
        // that leaves no room for the payload offset.
        assert_eq!(next_chunk(usize::MAX, 0), Some(usize::MAX));
        assert_eq!(next_chunk(usize::MAX, 2), None);
        assert_eq!(next_chunk(1, usize::MAX - 1), None);
    }

    /**
     * Tests that an `ANMF` too short to hold a `VP8L` header is walked past
     * rather than indexed into, which no capture can show because every
     * frame vips writes is longer than the header it declares. Works by
     * building the shortest chunk the walker accepts, at the two sub-chunk
     * tags that answer differently.
     * Input: a hand-built `ANMF` of exactly 24 bytes, once naming `VP8L`
     * and once naming `VP8 ` -> Output: no offset for the `VP8L` one,
     * because its `alpha_is_used` bit is not in the file, and an offset for
     * the `VP8 ` one, because a lossy frame has no alpha whatever follows.
     */
    #[test]
    fn a_frame_too_short_to_read_is_walked_past_rather_than_indexed_into() {
        // 24 bytes is the floor the walker accepts: a 16-byte frame header
        // and an 8-byte sub-chunk header, with no sub-chunk payload at all.
        // A `VP8L` signature and its `alpha_is_used` bit live in the five
        // bytes after that, which this file does not have.
        let truncated = |tag: &[u8; 4]| {
            let mut out = Vec::from(*b"RIFF\x20\x00\x00\x00WEBPANMF\x18\x00\x00\x00");
            out.extend_from_slice(&[0u8; 15]); // x, y, w, h, duration
            out.push(0); // frame info: blending on
            out.extend_from_slice(tag);
            out.extend_from_slice(&[0u8; 4]); // the sub-chunk's own size
            out
        };

        let lossless = truncated(b"VP8L");
        assert_eq!(lossless.len(), 12 + 8 + 24);
        assert!(
            opaque_blended_frame_offsets(&lossless).is_empty(),
            "a frame whose alpha bit is not in the file is not provably opaque"
        );
        assert!(matches!(
            disable_blending_on_opaque_frames(&lossless),
            Cow::Borrowed(_)
        ));

        // The lossy tag needs nothing past the header to answer, so the same
        // truncation still names it. That is the positive control: the
        // emptiness above is the length test firing, not the walk giving up.
        let lossy = truncated(b"VP8 ");
        assert_eq!(opaque_blended_frame_offsets(&lossy), vec![20 + 15]);
    }

    /**
     * Tests the escape hatch the read-only asymmetry points at, end to end:
     * two pages of an animated WebP, saved as an animated GIF, come back
     * with those two pages' delays on those two pages. This is also the
     * decisive argument for subsetting the delay array, because
     * `Raster::encode_gif` refuses an array whose length is not the page
     * count, so under vips's file-scoped rule this save could not happen at
     * all.
     * Input: `ANIM4_DELAY` at `page = 1, n = 2` -> Output: a two-frame GIF
     * that reads back as a 4x6 two-page roll with `delay` `[70, 200]`, the
     * centisecond rounding of 67 and 200, and `loop` 3.
     */
    #[test]
    fn two_pages_of_an_animation_save_as_a_two_frame_gif() {
        let roll = decode_webp_with(
            &ANIM4_DELAY,
            DecodeLimits::default(),
            LoadOptions::default().with_page(1).with_n(2),
        )
        .expect("frames 1 and 2 exist");
        assert_eq!(roll.pages_loaded(), 2);
        assert_eq!(roll.get_int_array("delay"), Some(&[67i64, 200][..]));

        // The refusal this avoids, spelled out: `encode_gif` reads `delay`
        // and requires one entry per page, so a four-entry array on a
        // two-page roll is an error rather than a silent mis-timing.
        let mut mistimed = roll.try_clone().expect("a 4x6 raster clones");
        mistimed.set_field("delay", MetadataValue::IntArray(vec![45, 67, 200, 12]));
        let message = mistimed
            .encode_gif(crate::gif::SaveOptions::default())
            .expect_err("four delays do not fit two pages")
            .to_string();
        assert!(message.contains("4 entries for 2 page"), "{message}");

        let bytes = roll
            .encode_gif(crate::gif::SaveOptions::default())
            .expect("a two-page roll writes a two-frame GIF");
        let back = crate::gif::decode_gif_with(
            &bytes,
            DecodeLimits::default(),
            crate::gif::LoadOptions::default().with_n(-1),
        )
        .expect("it reads back");
        assert_eq!((back.width(), back.height()), (4, 6));
        assert_eq!(back.pages_loaded(), 2);
        // 67 ms is 7 centiseconds on the GIF wire and comes back as 70; 200
        // survives exactly. The 10 ms floor `webpsave` applies is a WebP
        // save behaviour and does not reach this path.
        assert_eq!(back.get_int_array("delay"), Some(&[70i64, 200][..]));
        assert_eq!(back.get_field("loop"), Some(MetadataValue::Int(3)));
    }

    /**
     * Tests what saving an animation back as WebP actually does, which is
     * the read-only asymmetry made concrete: the roll comes out as one
     * tall still image, where `vips webpsave` on the same raster writes a
     * four-frame animation. Works by loading every page, encoding, and
     * loading the result.
     * Input: the four-page roll from `ANIM4_DELAY` -> Output: a 4x12
     * still whose pixels are the whole roll and which carries no
     * `n-pages`, no `delay` and no `loop`.
     */
    #[test]
    fn an_animation_saved_back_is_one_tall_still() {
        let roll = decode_webp_with(&ANIM4_DELAY, DecodeLimits::default(), all_pages())
            .expect("the four-frame capture decodes");
        assert_eq!(roll.pages_loaded(), 4);

        let bytes = roll
            .encode_webp(SaveOptions::default())
            .expect("a 4x12 Rgb8 raster encodes");
        // No `ANIM` chunk, because `image-webp` has no code that writes
        // one: the file is the plain `RIFF`/`WEBP`/`VP8L` form.
        assert_eq!(riff_chunks(&bytes), vec!["VP8L"]);

        let back = decode_webp(&bytes, DecodeLimits::default()).expect("it reads back");
        assert_eq!((back.width(), back.height()), (4, 12));
        assert_eq!(back.data(), &ANIM4_ROLL[..]);
        assert_eq!(back.pages_loaded(), 1);
        for field in ["n-pages", "page-height", "delay", "loop"] {
            assert_eq!(back.get_field(field), None, "{field} on the round trip");
        }
    }
}
