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
//! * **An animation loads frame 0 and says so.** A default `vips webpload`
//!   reads one frame and sets `n-pages` to the count in the *original*
//!   (`webp2vips.c:505-508`); the 4x9 toilet roll needs `[n=-1]`.
//!   [`decode_webp`] matches that default exactly. Refusing an animation
//!   would also be a regression, since frame 0 already decoded through the
//!   sniff route before this module owned the decode. Reading every frame
//!   is issue #569 and waits on the page model in #564.
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

use std::io::Cursor;
use std::path::Path;

use crate::codec::EncodeError;
use crate::imageio::{MetadataValue, SaveError};
use crate::pixel::PixelFormat;
use crate::raster::Raster;
use crate::source::{DeclaredGeometry, DecodeLimits, SourceError};

/// The largest width or height libwebp will encode
/// (`WEBP_MAX_DIMENSION`; `webpsave.c:740-742` rejects anything above it
/// with `image too large`).
///
/// `image-webp` 0.2.4 guards on `> 16384` rather than `> 16383`, one
/// pixel too generous, so a raster of exactly 16384 would encode into a
/// `VP8L` the reference decoder then refuses. libviprs applies this
/// ceiling instead of the crate's.
pub const MAX_DIMENSION: u32 = 16383;

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
/// Plain, `Default`, and module-scoped, so callers write
/// `webp::SaveOptions { compression, ..Default::default() }` and later
/// fields can be added without a breaking change.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
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
/// is what a default `vips webpload` does (`webp2vips.c:505-508`). Reading
/// every frame is issue #569 and needs the page model from #564; until then
/// `n-pages` is the signal that frames were left behind.
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
    let mut decoder = image_webp::WebPDecoder::new(Cursor::new(bytes)).map_err(decode_error)?;
    // Budget the metadata chunk reads before any of them run: `read_chunk`
    // refuses a chunk longer than this rather than allocating for it.
    decoder.set_memory_limit(usize::try_from(limits.max_alloc_bytes).unwrap_or(usize::MAX));

    let (width, height) = decoder.dimensions();
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
    let size = decoder
        .output_buffer_size()
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
    // The price is the decoder's `output_buffer_size` rather than a declared
    // product, so it goes through `check_alloc` with the geometry attached by
    // hand rather than through `check_image_alloc`, which would recompute it.
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
    let frames = decoder.is_animated().then(|| decoder.num_frames());

    let mut data = vec![0u8; size];
    decoder.read_image(&mut data).map_err(decode_error)?;

    let mut raster = Raster::new(width, height, format, data)?;
    for (field, blob) in chunks {
        if let Some(blob) = blob {
            raster.fields.set(field, MetadataValue::Blob(blob));
        }
    }
    if let Some(frames) = frames {
        raster.set_n_pages(frames);
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
            p += 8 + size + (size & 1);
        }
        out
    }
}
