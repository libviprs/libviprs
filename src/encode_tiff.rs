//! TIFF encode/decode lane.
//!
//! This module carries the TIFF surface the ported foreign and connection
//! cells reference: the `save_tiff` family of file writers on
//! [`Raster`], the in-memory `tiff_save` / `tiff_load` round-trip, and the
//! free [`tiff_page_count`] / [`decode_tiff_page`] readers for multi-page
//! files. It builds directly on the pure-Rust [`tiff`] crate so the encode
//! side can pick a compression per image and the decode side can walk the
//! IFD chain, neither of which the higher-level `image` facade exposes.
//!
//! ## Real versus deferred
//!
//! Everything the `tiff` crate can emit and read with a pure-Rust codec is
//! real:
//!
//! * [`Raster::save_tiff`] with [`TiffCompression::None`],
//!   [`TiffCompression::Lzw`], and [`TiffCompression::Deflate`].
//! * [`Raster::tiff_save`] / [`Raster::tiff_load`] (the streamed connection
//!   round-trip), which use Deflate so the trip stays lossless.
//! * [`tiff_page_count`] and [`decode_tiff_page`] over multi-page files.
//!
//! The remaining variants have no pure-Rust encoder in this build and return
//! a typed [`SaveError`] rather than panicking, so the call sites compile and
//! pin the error path:
//!
//! * [`Raster::save_tiff`] with [`TiffCompression::Jpeg`],
//!   [`TiffCompression::Ccitt`], or [`TiffCompression::Jp2k`] (JPEG-in-TIFF,
//!   CCITT G4 fax, and JPEG 2000 all need an external codec).
//! * [`Raster::save_bigtiff`] (64-bit-offset container, deferred).
//! * [`Raster::save_tiff_tiled`] (tiled layout plus pyramid/subifd, deferred).
//!
//! ## Page indexing
//!
//! `page` in [`decode_tiff_page`] is a **zero-based index**: page 0 is the
//! first image, and the valid range is `0..`[`tiff_page_count`]. That is the
//! libvips convention, whose `page` argument is `min: 0` on `tiffload`,
//! `pdfload`, `gifload`, `heifload` and `webpload` alike, and it is also the
//! [`tiff`] crate's, where `seek_to_image(0)` is the first IFD. A TIFF carries
//! no page numbers of its own: the IFD chain is a linked list, so there is
//! nothing in the file to number from one.
//!
//! `n-pages` is the matching **count**, not an index, so the last page of a
//! file is one less than the count. [`decode_tiff_page`] attaches it to every
//! raster it returns, single-page files included, which is what vips's
//! `tiffload` does (`vipsheader -f n-pages` reports 3 for a three-page file
//! and 1 for a one-page one).
//!
//! The uncapped way to read that count back is [`tiff_page_count`], **not**
//! [`Raster::get_n_pages`]. The accessor ports `vips_image_get_n_pages`
//! whole, sanity ceiling included, so it reports `1` for any stored count of
//! 10,000 or more no matter what the field really holds (issue #635). That is
//! reachable on this path rather than theoretical: [`DecodeLimits::max_pages`]
//! defaults to `100_000`, so a chain of 10,000 to 100,000 IFDs decodes
//! normally, attaches its real length, and reads back as a single page. The
//! stored value is untouched and still comes out of [`Raster::get_field`],
//! and [`tiff_page_count`] walks the chain itself and answers with the real
//! number. A `0..n` sweep stays in range either way, because the capped
//! answer is 1 rather than something longer than the file.
//!
//! The crate's PDF readers ([`crate::extract_page_image`] and friends) are
//! **1-based**, and that is deliberate rather than an oversight left behind
//! here. A PDF carries its own page numbering, [`crate::PdfInfo`] reports that
//! numbering straight out of the document, and the CLI exposes `--page` to
//! users on those terms. The rule across the crate is that a document's own
//! page number is 1-based and a position in a sequence of frames is 0-based.
//!
//! ## Resource limits
//!
//! Every entry point that reads untrusted bytes here takes a
//! [`DecodeLimits`], and the no-argument forms delegate to
//! [`DecodeLimits::default`] rather than running unbounded:
//! [`tiff_page_count_with_limits`], [`decode_tiff_page_with_limits`] and
//! [`Raster::tiff_load_with_limits`], behind [`tiff_page_count`],
//! [`decode_tiff_page`] and [`Raster::tiff_load`]. Four things on this path
//! are sized by the file and all four are bounded (issue #540):
//!
//! * The **file body**. `decode_tiff_page` reads the whole file rather than
//!   streaming it, because the vips multiband relabel below patches the bytes
//!   before the decoder sees them. That read goes through the crate's shared
//!   `read_file_bounded`, so it is capped at
//!   [`DecodeLimits::max_alloc_bytes`], checked against the declared length
//!   first and again against what was actually read. The same helper now
//!   backs [`crate::decode_file`]'s whole-file read (issue #629).
//! * The **relabel**. It patches the buffer the reader already owns instead of
//!   cloning it (`normalize_multiband_photometric_in_place`), so the peak
//!   footprint is one copy of the file, not two.
//! * The **IFD walk** that sources `n-pages`, capped at
//!   [`DecodeLimits::max_pages`] and stopped there rather than counted to the
//!   end. `tiff_page_count` streams and never reads the body at all. A *cyclic*
//!   chain is caught below this, by `tiff` 0.10.3's own union-find over the IFD
//!   edges; the ceiling is for the chain that is merely very long.
//! * The **pixel buffer**, through [`DecodeLimits::max_coord`] then
//!   [`DecodeLimits::max_pixels`] then an explicit
//!   `width * height * bands * bytes_per_sample` budget, all on the declared
//!   geometry and all before the buffer is reserved. Before #540 the result
//!   went straight to [`Raster::new`] and its 8 GiB
//!   `DEFAULT_MAX_ALLOC_BYTES`, sixteen times looser than the 512 MiB the
//!   crate publishes.
//!
//! The `tiff` crate's own `decoding_buffer_size` is tightened from
//! `max_alloc_bytes` too, and never loosened past its 256 MiB default, so the
//! effective ceiling on a page decode is the smaller of the two.
//!
//! ## Pixel formats
//!
//! The encoder handles the 8- and 16-bit gray, RGB, and RGBA formats, plus the
//! N-band [`PixelFormat::Multi8`] / [`PixelFormat::Multi16`] carriers: a
//! multiband uchar/ushort raster is written as a `BlackIsZero` base plus
//! `N - 1` unassociated-alpha extra samples, so it round-trips as a portable
//! integer carrier and vips reads the file back with the same band count and
//! samples. (A 4-band uchar/ushort raster is the named [`PixelFormat::Rgba8`] /
//! [`PixelFormat::Rgba16`] and travels the RGBA path.) See [`encode_multiband`]
//! for why the `BlackIsZero` layout is used in place of vips's RGB-plus-extra
//! layout for `>= 3` bands.
//!
//! On the decode side, a *foreign* `>= 5`-band raster that vips wrote is an RGB
//! photometric carrying `N - 3` extra samples, a layout the pure-Rust `tiff`
//! decoder rejects at every entry point (it funnels through `expand_chunk`,
//! which calls `colortype()` and errors for RGB-with-extra). [`decode_tiff_page`]
//! and [`Raster::tiff_load`] first relabel that file's `PhotometricInterpretation`
//! tag from RGB to `BlackIsZero` (see [`normalize_multiband_photometric`]); the
//! relabel never alters a sample byte, so the decoder then reads the N-band
//! raster as a `Multiband{N}` carrier with the exact samples vips stored.
//!
//! The float [`PixelFormat`] variants remain compute
//! intermediates with no TIFF representation here and surface a typed error.
//! Sixteen-bit samples are read from and written back to the raster buffer in
//! native byte order, matching the convention the rest of the crate uses.

use std::borrow::Cow;
use std::io::{Cursor, Read, Seek, Write};
use std::path::Path;

use tiff::ColorType;
use tiff::decoder::{Decoder, DecodingResult};
use tiff::encoder::{
    Compression, DeflateLevel, DirectoryEncoder, Predictor, TiffEncoder, TiffKind, colortype,
};
use tiff::tags::Tag;

use crate::codec::{DecodeError, TiffCompression};
use crate::imageio::SaveError;
use crate::pixel::PixelFormat;
use crate::raster::Raster;
use crate::sink::SinkError;
use crate::source::{DecodeLimits, read_file_bounded};

// ---------------------------------------------------------------------------
// Error helpers
//
// `save_tiff` and friends return the crate-wide `SaveError`, which the shared
// spine owns in `imageio.rs`. That type has no dedicated "unsupported
// compression" variant and this lane must not widen it, so the deferred paths
// reuse `SaveError::Encode(SinkError::Other(_))` with an accurate message
// (the alternative, `SaveError::UnsupportedExtension`, carries a tail
// enumerating the extensions this build can save, which would misreport a
// compression problem as a missing encoder). The decode helpers
// return `DecodeError` (an alias of `SourceError`) and wrap `tiff` crate
// failures as an image decoding error tagged with the TIFF format hint.
// ---------------------------------------------------------------------------

/// A deferred TIFF compression mode reported as a typed, matchable error.
fn unsupported_compression(mode: &str) -> SaveError {
    SaveError::Encode(SinkError::Other(format!(
        "TIFF {mode} compression is not supported by the pure-Rust encoder"
    )))
}

/// A pixel format with no TIFF representation in this build.
fn unsupported_pixels(format: PixelFormat) -> SaveError {
    SaveError::Encode(SinkError::Other(format!(
        "TIFF encoding of {format:?} pixels is not supported \
         (only 8- and 16-bit gray, RGB, and RGBA)"
    )))
}

/// A deferred TIFF container feature (BigTIFF, tiled layout) reported typed.
fn unsupported_feature(feature: &str) -> SaveError {
    SaveError::Encode(SinkError::Other(format!(
        "{feature} is not yet implemented in the pure-Rust TIFF lane"
    )))
}

/// Map a `tiff` crate encode failure onto the crate-wide save error.
fn tiff_encode_err(err: tiff::TiffError) -> SaveError {
    SaveError::Encode(SinkError::Other(format!("TIFF encode failed: {err}")))
}

/// Build a TIFF-tagged decode error from a free-form message.
fn decode_err(message: impl Into<String>) -> DecodeError {
    use image::error::{DecodingError, ImageFormatHint};
    DecodeError::Decode(image::ImageError::Decoding(DecodingError::new(
        ImageFormatHint::Exact(image::ImageFormat::Tiff),
        message.into(),
    )))
}

/// Map a `tiff` crate decode failure onto the crate-wide decode error.
fn tiff_decode_err(err: tiff::TiffError) -> DecodeError {
    decode_err(err.to_string())
}

// ---------------------------------------------------------------------------
// Encode
// ---------------------------------------------------------------------------

/// Reinterpret a native-byte-order 16-bit sample buffer as `u16` values.
fn as_u16_samples(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&c| u16::from_ne_bytes(c))
        .collect()
}

/// Encode this raster to in-memory TIFF bytes with the given compression.
///
/// LZW and Deflate apply the horizontal predictor, which is lossless and is
/// what makes those modes actually shrink natural images (and keeps this in
/// line with what libvips writes). `None` writes raw strips.
fn encode_to_vec(raster: &Raster, compression: TiffCompression) -> Result<Vec<u8>, SaveError> {
    // N-band multiband uchar/ushort rasters have no named TIFF colour type;
    // route them through the dedicated writer, which emits the vips tag layout
    // (Gray/RGB base + unassociated-alpha extra samples). The named 1/3/4-band
    // formats fall through to the standard `write_image` path below.
    match raster.format() {
        PixelFormat::Multi8(n) => {
            return encode_multiband(raster, compression, n.get() as usize, 1);
        }
        PixelFormat::Multi16(n) => {
            return encode_multiband(raster, compression, n.get() as usize, 2);
        }
        _ => {}
    }

    let (comp, predictor) = match compression {
        TiffCompression::None => (Compression::Uncompressed, Predictor::None),
        TiffCompression::Lzw => (Compression::Lzw, Predictor::Horizontal),
        TiffCompression::Deflate => (
            Compression::Deflate(DeflateLevel::default()),
            Predictor::Horizontal,
        ),
        TiffCompression::Jpeg => return Err(unsupported_compression("JPEG")),
        TiffCompression::Ccitt => return Err(unsupported_compression("CCITT G4")),
        TiffCompression::Jp2k => return Err(unsupported_compression("JPEG 2000")),
    };

    let width = raster.width();
    let height = raster.height();
    let mut buf = Vec::new();
    {
        let mut encoder = TiffEncoder::new(Cursor::new(&mut buf))
            .map_err(tiff_encode_err)?
            .with_compression(comp)
            .with_predictor(predictor);

        let write_result = match raster.format() {
            PixelFormat::Gray8 => {
                encoder.write_image::<colortype::Gray8>(width, height, raster.data())
            }
            PixelFormat::Rgb8 => {
                encoder.write_image::<colortype::RGB8>(width, height, raster.data())
            }
            PixelFormat::Rgba8 => {
                encoder.write_image::<colortype::RGBA8>(width, height, raster.data())
            }
            PixelFormat::Gray16 => {
                let samples = as_u16_samples(raster.data());
                encoder.write_image::<colortype::Gray16>(width, height, &samples)
            }
            PixelFormat::Rgb16 => {
                let samples = as_u16_samples(raster.data());
                encoder.write_image::<colortype::RGB16>(width, height, &samples)
            }
            PixelFormat::Rgba16 => {
                let samples = as_u16_samples(raster.data());
                encoder.write_image::<colortype::RGBA16>(width, height, &samples)
            }
            other => return Err(unsupported_pixels(other)),
        };
        write_result.map_err(tiff_encode_err)?;
    }
    Ok(buf)
}

/// Encode an N-band multiband raster as a portable, round-trippable TIFF.
///
/// vips has no "multiband" TIFF colour type: `tiffsave` emits an N-band
/// uchar/ushort raster as a `BlackIsZero` base (`<= 2` bands) or an RGB base
/// (`>= 3` bands) plus `N - base` unassociated-alpha extra samples. We write
/// the `BlackIsZero`-plus-extra layout for *every* multiband count, which
/// carries the identical pixel samples and band count (so vips reads it back
/// with the same values) and, unlike vips's RGB-plus-extra layout for `>= 5`
/// bands, is one the pure-Rust `tiff` decoder can read — `read_image` rejects
/// an RGB photometric carrying extra samples, so an RGB layout would not
/// round-trip within this crate. That load-bearing external invariant (the
/// `tiff` crate accepting `BlackIsZero`+extra but rejecting RGB+extra) is pinned
/// by `decode_rgb_plus_extra_like_vips_round_trips` so a crate upgrade that
/// changed it would fail loudly rather than silently. The one difference from
/// vips is the reported
/// interpretation tag (`b-w`/`multiband` here vs `srgb` for vips's `>= 3` band
/// files); the carried integer samples are byte-identical.
///
/// The pixel buffer is handed to the `tiff` crate as a single wide "gray"
/// image of `width * channels` columns, then the directory tags are overridden
/// (`ImageWidth`, `SamplesPerPixel`, `BitsPerSample`, `SampleFormat`,
/// `ExtraSamples`) to describe the true N-band geometry. That trick keeps the
/// strip bytes identical to a real N-band write only when no per-sample
/// predictor is applied, so [`Predictor::None`] is forced here (the
/// compression itself stays lossless).
fn encode_multiband(
    raster: &Raster,
    compression: TiffCompression,
    channels: usize,
    depth_bytes: usize,
) -> Result<Vec<u8>, SaveError> {
    let comp = match compression {
        TiffCompression::None => Compression::Uncompressed,
        TiffCompression::Lzw => Compression::Lzw,
        TiffCompression::Deflate => Compression::Deflate(DeflateLevel::default()),
        TiffCompression::Jpeg => return Err(unsupported_compression("JPEG")),
        TiffCompression::Ccitt => return Err(unsupported_compression("CCITT G4")),
        TiffCompression::Jp2k => return Err(unsupported_compression("JPEG 2000")),
    };
    let width = raster.width();
    let height = raster.height();
    let wide = u32::try_from(width as usize * channels).map_err(|_| {
        SaveError::Encode(SinkError::Other(
            "multiband TIFF width * bands exceeds the 32-bit TIFF dimension limit".to_string(),
        ))
    })?;

    let mut buf = Vec::new();
    {
        let mut encoder = TiffEncoder::new(Cursor::new(&mut buf))
            .map_err(tiff_encode_err)?
            .with_compression(comp)
            .with_predictor(Predictor::None);
        match depth_bytes {
            1 => {
                let mut image = encoder
                    .new_image::<colortype::Gray8>(wide, height)
                    .map_err(tiff_encode_err)?;
                write_multiband_tags(image.encoder(), width, channels, 8)
                    .map_err(tiff_encode_err)?;
                image.write_data(raster.data()).map_err(tiff_encode_err)?;
            }
            _ => {
                let samples = as_u16_samples(raster.data());
                let mut image = encoder
                    .new_image::<colortype::Gray16>(wide, height)
                    .map_err(tiff_encode_err)?;
                write_multiband_tags(image.encoder(), width, channels, 16)
                    .map_err(tiff_encode_err)?;
                image.write_data(&samples).map_err(tiff_encode_err)?;
            }
        }
    }
    Ok(buf)
}

/// Override the base-image directory tags to describe an N-band `BlackIsZero`
/// raster with `channels - 1` unassociated-alpha extra samples. Re-writing a
/// tag replaces the earlier entry in the directory, so this corrects the
/// geometry the single-band base write recorded. The base `new_image` already
/// wrote `PhotometricInterpretation = BlackIsZero`, which is kept.
fn write_multiband_tags<W: Write + Seek, K: TiffKind>(
    dir: &mut DirectoryEncoder<'_, W, K>,
    width: u32,
    channels: usize,
    bits: u16,
) -> tiff::TiffResult<()> {
    dir.write_tag(Tag::ImageWidth, width)?;
    dir.write_tag(Tag::SamplesPerPixel, u16::try_from(channels)?)?;
    dir.write_tag(Tag::BitsPerSample, &vec![bits; channels][..])?;
    // SampleFormat 1 = unsigned integer, one entry per sample.
    dir.write_tag(Tag::SampleFormat, &vec![1u16; channels][..])?;
    // ExtraSamples 2 = unassociated alpha, one per non-colour sample, as vips
    // writes. The single colour sample is the BlackIsZero base.
    dir.write_tag(Tag::ExtraSamples, &vec![2u16; channels - 1][..])?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

/// Turn a channel count and the `tiff` decoder's sample buffer into a
/// canonical [`PixelFormat`] and native-byte-order pixel buffer.
fn interpret(
    channels: usize,
    result: DecodingResult,
) -> Result<(PixelFormat, Vec<u8>), DecodeError> {
    match result {
        DecodingResult::U8(values) => {
            let format = PixelFormat::with_channels(channels, 1)
                .ok_or_else(|| decode_err("unsupported TIFF channel count"))?;
            Ok((format, values))
        }
        DecodingResult::U16(values) => {
            let format = PixelFormat::with_channels(channels, 2)
                .ok_or_else(|| decode_err("unsupported TIFF channel count"))?;
            let bytes = values.into_iter().flat_map(u16::to_ne_bytes).collect();
            Ok((format, bytes))
        }
        _ => Err(decode_err(
            "unsupported TIFF sample format \
             (only 8- and 16-bit unsigned samples are decoded here)",
        )),
    }
}

/// Resolve the channel count and the per-sample bit depth of the current
/// image.
///
/// `colortype()` maps the common photometrics and the `BlackIsZero`-with-extra
/// Multiband case. A vips-written `>= 5`-band raster is an RGB photometric with
/// extra samples, which `colortype()` rejects — but by the time the decoder
/// reaches here that file has already been relabelled to `BlackIsZero` by
/// [`normalize_multiband_photometric`], so it resolves as `Multiband{N}`. Any
/// residual colour type the decoder still refuses (e.g. CMYK-with-extra, which
/// vips does not write for these rasters) surfaces as a typed error rather than
/// a silent, wrong channel count.
///
/// The depth rides along out of the same `colortype()` call because the
/// pre-decode allocation budget needs the product: a pixel count on its own
/// sees neither the band count nor the sample depth, so `max_pixels` alone
/// cannot bound the buffer a page decodes into.
fn resolve_channels_and_depth<R: Read + Seek>(
    decoder: &mut Decoder<R>,
) -> Result<(usize, u8), DecodeError> {
    match decoder.colortype().map_err(tiff_decode_err)? {
        ColorType::Gray(bits) => Ok((1, bits)),
        ColorType::RGB(bits) => Ok((3, bits)),
        ColorType::RGBA(bits) => Ok((4, bits)),
        ColorType::Multiband {
            num_samples,
            bit_depth,
        } => Ok((num_samples as usize, bit_depth)),
        other => Err(decode_err(format!(
            "unsupported TIFF color type {other:?} \
             (gray, RGB, RGBA, and multiband uchar/ushort are decoded here)"
        ))),
    }
}

/// Relabel a vips-style RGB-plus-extra-samples raster so the pure-Rust decoder
/// can read it.
///
/// vips writes an N-band (`>= 5`) uchar/ushort raster as an RGB photometric base
/// plus `N - 3` unassociated-alpha extra samples. The `tiff` crate rejects an
/// RGB (or CMYK/YCbCr) photometric carrying extra samples at *every* decode
/// entry point — `read_image`, `read_chunk`, and `read_chunk_bytes` all funnel
/// through `expand_chunk`, which begins by calling `colortype()` and errors for
/// that layout — so the file cannot be read as-is. Rewriting the
/// `PhotometricInterpretation` tag (262) in place from RGB (2) to `BlackIsZero`
/// (1) makes the decoder resolve the samples as a `Multiband{N}` raster. That is
/// a pure relabel: the decoder applies a colour transform only for `WhiteIsZero`
/// and none for `BlackIsZero`/`Multiband`, so the recovered interleaved N-band
/// buffer is byte-identical to the samples vips stored.
///
/// The rewrite is confined to classic-TIFF IFDs whose `PhotometricInterpretation`
/// is RGB (2) and whose `SamplesPerPixel` is `> 4` (3 = RGB and 4 = RGBA are read
/// natively and left untouched). Malformed input, BigTIFF, and non-matching IFDs
/// are left exactly as-is so the normal decoder reports its own typed error; all
/// reads are bounds-checked and the IFD walk is iteration-bounded, so hostile or
/// truncated bytes can neither panic nor loop.
fn photometric_patch_offsets(bytes: &[u8]) -> Vec<usize> {
    let little_endian = match bytes.get(0..2) {
        Some(b"II") => true,
        Some(b"MM") => false,
        _ => return Vec::new(),
    };
    let read_u16 = |offset: usize| -> Option<u16> {
        let s = bytes.get(offset..offset + 2)?;
        let b = [s[0], s[1]];
        Some(if little_endian {
            u16::from_le_bytes(b)
        } else {
            u16::from_be_bytes(b)
        })
    };
    let read_u32 = |offset: usize| -> Option<u32> {
        let s = bytes.get(offset..offset + 4)?;
        let b = [s[0], s[1], s[2], s[3]];
        Some(if little_endian {
            u32::from_le_bytes(b)
        } else {
            u32::from_be_bytes(b)
        })
    };

    // Classic TIFF (magic 42) only; a BigTIFF (43) header has a different IFD
    // layout, and vips does not write these rasters as BigTIFF, so leave it.
    if read_u16(2) != Some(42) {
        return Vec::new();
    }
    let mut ifd_offset = match read_u32(4) {
        Some(o) => o as usize,
        None => return Vec::new(),
    };

    let mut offsets = Vec::new();
    // Bound the IFD walk so a malformed next-IFD pointer cannot spin forever.
    let mut guard = 0u32;
    while ifd_offset != 0 && guard < 1 << 16 {
        guard += 1;
        let count = match read_u16(ifd_offset) {
            Some(c) => c as usize,
            None => break,
        };
        let entries = ifd_offset + 2;
        // PhotometricInterpretation (262) and SamplesPerPixel (277) are both
        // SHORT/count-1 tags whose value sits inline in the first two bytes of
        // the entry's 12-byte value/offset field (offset + 8).
        let mut photometric = None;
        let mut photometric_value_off = None;
        let mut samples_per_pixel = None;
        for i in 0..count {
            let entry = entries + i * 12;
            let value_off = entry + 8;
            match read_u16(entry) {
                Some(262) => {
                    photometric = read_u16(value_off);
                    photometric_value_off = Some(value_off);
                }
                Some(277) => samples_per_pixel = read_u16(value_off),
                Some(_) => {}
                None => break,
            }
        }
        if photometric == Some(2)
            && samples_per_pixel.is_some_and(|n| n > 4)
            && let Some(off) = photometric_value_off
        {
            offsets.push(off);
        }
        ifd_offset = match read_u32(entries + count * 12) {
            Some(o) => o as usize,
            None => break,
        };
    }
    offsets
}

/// Return the TIFF bytes with any vips RGB-plus-extra `>= 5`-band raster relabelled
/// to `BlackIsZero` (see [`photometric_patch_offsets`]). Borrows the input
/// unchanged when there is nothing to rewrite, cloning only when a patch applies.
fn normalize_multiband_photometric(bytes: &[u8]) -> Cow<'_, [u8]> {
    let offsets = photometric_patch_offsets(bytes);
    if offsets.is_empty() {
        return Cow::Borrowed(bytes);
    }
    let mut owned = bytes.to_vec();
    apply_photometric_patches(&mut owned, &offsets);
    Cow::Owned(owned)
}

/// As [`normalize_multiband_photometric`], but rewriting a buffer the caller
/// already owns. The `Cow` form has to clone the whole file to patch it, which
/// doubles the peak footprint of a decode; a caller holding its own `Vec` (the
/// page readers, which read the file in themselves) patches it where it lies
/// and never pays that.
fn normalize_multiband_photometric_in_place(bytes: &mut [u8]) {
    let offsets = photometric_patch_offsets(bytes);
    apply_photometric_patches(bytes, &offsets);
}

/// Write `BlackIsZero` (1) into each `PhotometricInterpretation` value slot
/// [`photometric_patch_offsets`] found. Every write is bounds-checked, so a
/// truncated file is left alone rather than panicking.
fn apply_photometric_patches(bytes: &mut [u8], offsets: &[usize]) {
    if offsets.is_empty() {
        return;
    }
    let little_endian = bytes.first() == Some(&b'I');
    let value = if little_endian {
        1u16.to_le_bytes()
    } else {
        1u16.to_be_bytes()
    };
    for &off in offsets {
        if let Some(slot) = bytes.get_mut(off..off + 2) {
            slot.copy_from_slice(&value);
        }
    }
}

/// Read the EXIF-style orientation from the TIFF `Orientation` tag (274),
/// which vips writes when saving an oriented raster. An absent or
/// out-of-range tag reads as `1` (upright), matching vips.
fn read_tiff_orientation<R: Read + Seek>(decoder: &mut Decoder<R>) -> u8 {
    match decoder.find_tag_unsigned::<u16>(Tag::Orientation) {
        Ok(Some(v)) if (1..=8).contains(&v) => v as u8,
        _ => 1,
    }
}

/// The `tiff` crate's own default ceiling on a single `DecodingResult`, from
/// `tiff::decoder::Limits::default()`. libviprs only ever tightens it from
/// [`DecodeLimits::max_alloc_bytes`], never loosens it, so the effective bound
/// on a page decode is the smaller of the two. Raising `max_alloc_bytes` above
/// this therefore does not raise the effective ceiling here; lowering it below
/// does lower it.
const TIFF_DECODING_BUFFER_DEFAULT: u64 = 256 * 1024 * 1024;

/// Open a decoder over `reader` with libviprs's allocation budget pushed into
/// the `tiff` crate, so the crate refuses an oversized internal buffer itself
/// rather than relying on the caller's own pre-decode check alone.
fn open_decoder<R: Read + Seek>(
    reader: R,
    limits: DecodeLimits,
) -> Result<Decoder<R>, DecodeError> {
    let budget = limits.max_alloc_bytes.min(TIFF_DECODING_BUFFER_DEFAULT);
    let budget = usize::try_from(budget).unwrap_or(usize::MAX);
    let mut tiff_limits = tiff::decoder::Limits::default();
    tiff_limits.decoding_buffer_size = budget;
    tiff_limits.intermediate_buffer_size = tiff_limits.intermediate_buffer_size.min(budget);
    tiff_limits.ifd_value_size = tiff_limits.ifd_value_size.min(budget);
    Ok(Decoder::new(reader)
        .map_err(tiff_decode_err)?
        .with_limits(tiff_limits))
}

/// Read the decoder's currently-selected image into a [`Raster`], applying
/// `limits` to the declared geometry before anything is allocated for it.
///
/// The order is the one the rest of the crate uses:
/// [`DecodeLimits::max_coord`] on each axis, then
/// [`DecodeLimits::max_pixels`] on the product, then the explicit
/// `width * height * bands * bytes_per_sample` allocation budget, which is the
/// only one of the three that can see the band count and the sample depth.
/// All three run on what the IFD *declares*, so a decompression bomb is
/// refused before its buffer is reserved rather than after.
fn decode_current_image<R: Read + Seek>(
    decoder: &mut Decoder<R>,
    limits: DecodeLimits,
) -> Result<Raster, DecodeError> {
    let (width, height) = decoder.dimensions().map_err(tiff_decode_err)?;
    limits.check_coord(width, height)?;
    limits.check_pixels(width, height)?;
    let (channels, bit_depth) = resolve_channels_and_depth(decoder)?;
    let needed = u64::from(width)
        .saturating_mul(u64::from(height))
        .saturating_mul(channels as u64)
        .saturating_mul(u64::from(bit_depth).div_ceil(8));
    limits.check_alloc("TIFF page pixel buffer", needed)?;
    let orientation = read_tiff_orientation(decoder);
    let result = decoder.read_image().map_err(tiff_decode_err)?;
    let (format, data) = interpret(channels, result)?;
    let mut raster = Raster::new(width, height, format, data)?;
    raster.meta.orientation = orientation;
    Ok(raster)
}

/// Count the pages (IFDs) in a multi-page TIFF file, under the default
/// [`DecodeLimits`].
///
/// Equivalent to [`tiff_page_count_with_limits`] with
/// [`DecodeLimits::default`], whose [`DecodeLimits::max_pages`] is `100_000`.
///
/// # Errors
///
/// Returns [`DecodeError`] if the file cannot be opened or the TIFF header or
/// its IFD chain is malformed, or [`DecodeError::PageLimitExceeded`] if the
/// chain runs past the default page ceiling.
pub fn tiff_page_count(path: &Path) -> Result<u32, DecodeError> {
    tiff_page_count_with_limits(path, DecodeLimits::default())
}

/// Count the pages (IFDs) in a multi-page TIFF file, bounding the IFD walk
/// with [`DecodeLimits::max_pages`].
///
/// A TIFF's pages are a linked list with no count in the header, so the only
/// way to report one is to walk the chain. `max_pages` is what keeps that walk
/// from being unbounded work on a hostile file: it stops and reports
/// [`DecodeError::PageLimitExceeded`] the moment it reaches the ceiling,
/// rather than running to the end to find out how far past it the file goes.
/// The file is streamed rather than read into memory, so nothing here is sized
/// by the input.
///
/// # Errors
///
/// Returns [`DecodeError::Io`] if the file cannot be opened,
/// [`DecodeError::PageLimitExceeded`] if the chain declares more than
/// [`DecodeLimits::max_pages`] pages, or [`DecodeError`] if the TIFF header or
/// its IFD chain is malformed.
pub fn tiff_page_count_with_limits(path: &Path, limits: DecodeLimits) -> Result<u32, DecodeError> {
    let file = std::fs::File::open(path)?;
    let mut decoder = open_decoder(std::io::BufReader::new(file), limits)?;
    count_images(&mut decoder, limits.max_pages)
}

/// Walk a decoder's IFD chain and report how many images it holds, giving up
/// with [`DecodeError::PageLimitExceeded`] once `max_pages` is reached.
///
/// The decoder is left positioned on the last image it read. Seeking back with
/// `seek_to_image` afterwards is cheap, because the walk fills the decoder's
/// own IFD offset table, so a caller that wants both the count and one page's
/// pixels does not need a second decoder over the same bytes.
///
/// A *cyclic* chain is caught underneath rather than here: `tiff` 0.10.3 runs
/// union-find over the IFD edges (`decoder::cycles::IfdCycles`) and returns
/// `CycleInOffsets` on a back edge. The ceiling is for the chain that is
/// merely very long, which nothing below bounds.
fn count_images<R: Read + Seek>(
    decoder: &mut Decoder<R>,
    max_pages: u32,
) -> Result<u32, DecodeError> {
    let mut count = 1u32;
    while decoder.more_images() {
        if count >= max_pages {
            return Err(DecodeError::PageLimitExceeded { max_pages });
        }
        decoder.next_image().map_err(tiff_decode_err)?;
        count = count.saturating_add(1);
    }
    Ok(count)
}

/// Decode a single page from a multi-page TIFF file, indexed from zero.
///
/// `page` is a **zero-based index** into the file's IFD chain: page 0 is the
/// first image and the valid range is `0..`[`tiff_page_count`]. This mirrors
/// the libvips `page` argument, which is `min: 0` on `tiffload` and on every
/// other multi-page loader vips has.
///
/// The returned raster carries `n-pages` set to the file's page count, so the
/// bound `page` has to stay under travels back with the pixels. It is a count
/// and `page` is an index, so the last page of a file is one less than it.
///
/// Read it back with [`tiff_page_count`] rather than with
/// [`Raster::get_n_pages`] if the chain may be long. The accessor ports
/// `vips_image_get_n_pages`'s sanity check, which reports a single page for
/// any stored count of 10,000 or more (issue #635), while
/// [`DecodeLimits::max_pages`] lets a chain run to `100_000`. The attached
/// field keeps the real number and [`Raster::get_field`] still hands it over;
/// only the accessor caps. Sweeping `0..get_n_pages()` never runs off the
/// end, because the capped answer is 1.
///
/// PDF page numbers in this crate are 1-based instead
/// ([`crate::extract_page_image`] and friends); see the module docs for why
/// the two conventions differ.
///
/// # Errors
///
/// Returns [`DecodeError`] if the file cannot be opened, `page` is at or past
/// the file's page count, the TIFF is malformed, or the page uses a color or
/// sample type this lane does not decode. Under the default
/// [`DecodeLimits`] it also returns the limit variants
/// [`decode_tiff_page_with_limits`] documents.
pub fn decode_tiff_page(path: &Path, page: u32) -> Result<Raster, DecodeError> {
    decode_tiff_page_with_limits(path, page, DecodeLimits::default())
}

/// As [`decode_tiff_page`], with the resource ceilings the decode runs under
/// given explicitly instead of taken from [`DecodeLimits::default`].
///
/// Everything sized by the file is bounded here, in this order:
///
/// 1. The file body, read in whole because the vips multiband relabel patches
///    the bytes before the decoder sees them, is capped at
///    [`DecodeLimits::max_alloc_bytes`] and patched in place rather than
///    cloned.
/// 2. The IFD walk that sources `n-pages` is capped at
///    [`DecodeLimits::max_pages`], and stops there rather than counting to the
///    end of the chain.
/// 3. The page's declared geometry goes through
///    [`DecodeLimits::max_coord`], then [`DecodeLimits::max_pixels`], then an
///    explicit `width * height * bands * bytes_per_sample` check against
///    [`DecodeLimits::max_alloc_bytes`], all before the pixel buffer is
///    reserved.
///
/// The `tiff` crate's own `decoding_buffer_size` is tightened from
/// `max_alloc_bytes` too, never loosened past its 256 MiB default.
///
/// # Errors
///
/// * [`DecodeError::Io`] if the file cannot be opened or read.
/// * [`DecodeError::AllocLimitExceeded`] if the file body or the page's pixel
///   buffer would be larger than [`DecodeLimits::max_alloc_bytes`].
/// * [`DecodeError::PageLimitExceeded`] if the IFD chain declares more than
///   [`DecodeLimits::max_pages`] pages.
/// * [`DecodeError::CoordLimitExceeded`] if either declared axis exceeds
///   [`DecodeLimits::max_coord`], or
///   [`DecodeError::DimensionLimitExceeded`] if `width * height` exceeds
///   [`DecodeLimits::max_pixels`].
/// * [`DecodeError`] if `page` is at or past the file's page count, the TIFF
///   is malformed, or the page uses a color or sample type this lane does not
///   decode.
pub fn decode_tiff_page_with_limits(
    path: &Path,
    page: u32,
    limits: DecodeLimits,
) -> Result<Raster, DecodeError> {
    let mut bytes = read_file_bounded(path, limits, "TIFF file body")?;
    normalize_multiband_photometric_in_place(&mut bytes);

    // Counting first buys two things: an out-of-range index reports the bound
    // it missed instead of whatever the `tiff` crate's seek happens to say,
    // and the count can ride back out on the raster as `n-pages`. The walk
    // fills the decoder's IFD offset table, so seeking back to `page` on the
    // same decoder is one directory read rather than a second walk.
    let mut decoder = open_decoder(Cursor::new(bytes.as_slice()), limits)?;
    let n_pages = count_images(&mut decoder, limits.max_pages)?;
    if page >= n_pages {
        return Err(decode_err(format!(
            "TIFF page {page} is out of range: pages are indexed from 0 and \
             this file has {n_pages}"
        )));
    }

    decoder
        .seek_to_image(page as usize)
        .map_err(tiff_decode_err)?;
    let mut raster = decode_current_image(&mut decoder, limits)?;
    raster.set_n_pages(n_pages);
    Ok(raster)
}

// ---------------------------------------------------------------------------
// Raster methods
// ---------------------------------------------------------------------------

impl Raster {
    /// Save as a single-page TIFF, choosing the strip compression.
    ///
    /// [`TiffCompression::None`], [`TiffCompression::Lzw`], and
    /// [`TiffCompression::Deflate`] are written by the pure-Rust encoder (LZW
    /// and Deflate are lossless and use the horizontal predictor). The
    /// external-codec modes ([`TiffCompression::Jpeg`],
    /// [`TiffCompression::Ccitt`], [`TiffCompression::Jp2k`]) are not
    /// available in this build.
    ///
    /// # Errors
    ///
    /// Returns [`SaveError::Encode`] if the compression mode or the raster's
    /// pixel format has no pure-Rust TIFF encoder, if the `tiff` encoder
    /// rejects the data, or [`SaveError::Io`] if writing the file fails.
    pub fn save_tiff(&self, path: &Path, compression: TiffCompression) -> Result<(), SaveError> {
        let bytes = encode_to_vec(self, compression)?;
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Save as a BigTIFF (64-bit offsets, for files larger than 4 GB).
    ///
    /// # Errors
    ///
    /// BigTIFF encoding is deferred, so this always returns a typed
    /// [`SaveError::Encode`] naming the unimplemented feature.
    pub fn save_bigtiff(&self, path: &Path, compression: TiffCompression) -> Result<(), SaveError> {
        let _ = (path, compression);
        Err(unsupported_feature("BigTIFF encoding"))
    }

    /// Save as a tiled TIFF, optionally with a pyramid and SubIFDs.
    ///
    /// # Errors
    ///
    /// Tiled TIFF encoding (and the pyramid/subifd layout it feeds) is
    /// deferred, so this always returns a typed [`SaveError::Encode`] naming
    /// the unimplemented feature.
    pub fn save_tiff_tiled(
        &self,
        path: &Path,
        compression: TiffCompression,
        tile_width: u32,
        tile_height: u32,
        pyramid: bool,
        subifd: bool,
    ) -> Result<(), SaveError> {
        let _ = (path, compression, tile_width, tile_height, pyramid, subifd);
        Err(unsupported_feature("Tiled TIFF encoding"))
    }

    /// Encode this raster as in-memory TIFF bytes (the streamed connection
    /// save). Uses Deflate so the round-trip through [`Raster::tiff_load`]
    /// stays lossless.
    ///
    /// The contract is infallible: the 8- and 16-bit gray/RGB/RGBA and N-band
    /// multiband formats always encode, and encoding into memory does no
    /// fallible I/O. A float raster (a compute intermediate with no TIFF form)
    /// yields an empty buffer rather than a spurious error.
    pub fn tiff_save(&self) -> Vec<u8> {
        match encode_to_vec(self, TiffCompression::Deflate) {
            Ok(bytes) => bytes,
            Err(_err) => {
                // An empty save is otherwise silent and undiagnosable, so I
                // surface the reason in debug builds. The contract stays
                // infallible: float rasters have no TIFF form and legitimately
                // yield an empty buffer here.
                #[cfg(debug_assertions)]
                eprintln!("tiff_save: encoding produced no bytes ({_err})");
                Vec::new()
            }
        }
    }

    /// Decode a raster from in-memory TIFF bytes (the inverse of
    /// [`Raster::tiff_save`]); reads the first page.
    ///
    /// Equivalent to [`Raster::tiff_load_with_limits`] with
    /// [`DecodeLimits::default`]. The caller already holds the bytes, so
    /// nothing here reads a file, but the *decoded* geometry is bounded the
    /// same way [`decode_tiff_page`] bounds it.
    ///
    /// # Errors
    ///
    /// Returns [`DecodeError`] if the bytes are not a valid TIFF, or the
    /// first page uses a color or sample type this lane does not decode, plus
    /// the limit variants [`Raster::tiff_load_with_limits`] documents.
    pub fn tiff_load(data: &[u8]) -> Result<Raster, DecodeError> {
        Self::tiff_load_with_limits(data, DecodeLimits::default())
    }

    /// As [`Raster::tiff_load`], with the resource ceilings given explicitly.
    ///
    /// The first page's declared geometry goes through
    /// [`DecodeLimits::max_coord`], [`DecodeLimits::max_pixels`] and the
    /// explicit `max_alloc_bytes` buffer budget before the pixel buffer is
    /// reserved. There is no page walk (only page 0 is read) and no file read,
    /// so [`DecodeLimits::max_pages`] does not apply.
    ///
    /// # Errors
    ///
    /// * [`DecodeError::CoordLimitExceeded`] if either declared axis exceeds
    ///   [`DecodeLimits::max_coord`].
    /// * [`DecodeError::DimensionLimitExceeded`] if `width * height` exceeds
    ///   [`DecodeLimits::max_pixels`].
    /// * [`DecodeError::AllocLimitExceeded`] if the pixel buffer would be
    ///   larger than [`DecodeLimits::max_alloc_bytes`].
    /// * [`DecodeError`] if the bytes are not a valid TIFF, or the first page
    ///   uses a color or sample type this lane does not decode.
    pub fn tiff_load_with_limits(data: &[u8], limits: DecodeLimits) -> Result<Raster, DecodeError> {
        let data = normalize_multiband_photometric(data);
        let mut decoder = open_decoder(Cursor::new(data.as_ref()), limits)?;
        decode_current_image(&mut decoder, limits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::decode_file;
    use tiff::encoder::{TiffEncoder, colortype};

    fn ramp_gray8(w: u32, h: u32) -> Raster {
        let data: Vec<u8> = (0..w * h).map(|i| (i % w) as u8).collect();
        Raster::new(w, h, PixelFormat::Gray8, data).unwrap()
    }

    fn ramp_gray16(w: u32, h: u32) -> Raster {
        let samples: Vec<u16> = (0..w * h)
            .map(|i| ((i % w) as u16).saturating_mul(200))
            .collect();
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_ne_bytes()).collect();
        Raster::new(w, h, PixelFormat::Gray16, bytes).unwrap()
    }

    fn ramp_rgb8(w: u32, h: u32) -> Raster {
        let data: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                let x = (i % w) as u8;
                let y = (i / w) as u8;
                [x, y, x.wrapping_add(y)]
            })
            .collect();
        Raster::new(w, h, PixelFormat::Rgb8, data).unwrap()
    }

    fn ramp_rgba16(w: u32, h: u32) -> Raster {
        let samples: Vec<u16> = (0..w * h)
            .flat_map(|i| {
                let x = (i % w) as u16;
                let y = (i / w) as u16;
                [
                    x.saturating_mul(200),
                    y.saturating_mul(200),
                    x.saturating_add(y).saturating_mul(100),
                    u16::MAX,
                ]
            })
            .collect();
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_ne_bytes()).collect();
        Raster::new(w, h, PixelFormat::Rgba16, bytes).unwrap()
    }

    fn multipage_gray8_fixture(pages: u8) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let mut encoder = TiffEncoder::new(Cursor::new(&mut buf)).unwrap();
            for p in 0..pages {
                let data = vec![p.saturating_mul(10); 16];
                encoder
                    .write_image::<colortype::Gray8>(4, 4, &data)
                    .unwrap();
            }
        }
        buf
    }

    #[test]
    fn save_tiff_none_round_trips_bit_exact() {
        let im = ramp_gray8(64, 64);
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("none.tif");
        im.save_tiff(&out, TiffCompression::None).unwrap();
        let back = decode_file(&out).unwrap();
        assert_eq!(back.width(), im.width());
        assert_eq!(back.height(), im.height());
        assert_eq!(back.data(), im.data());
    }

    #[test]
    fn save_tiff_deflate_round_trips_bit_exact() {
        let im = ramp_gray8(64, 64);
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("deflate.tif");
        im.save_tiff(&out, TiffCompression::Deflate).unwrap();
        let back = decode_file(&out).unwrap();
        assert_eq!(back.width(), im.width());
        assert_eq!(back.height(), im.height());
        assert_eq!(back.data(), im.data(), "Deflate TIFF must be lossless");
    }

    #[test]
    fn save_tiff_lzw_smaller_than_none() {
        let im = ramp_gray8(128, 128);
        let dir = tempfile::tempdir().unwrap();
        let out_lzw = dir.path().join("lzw.tif");
        let out_none = dir.path().join("none.tif");
        im.save_tiff(&out_lzw, TiffCompression::Lzw).unwrap();
        im.save_tiff(&out_none, TiffCompression::None).unwrap();

        let lzw_size = std::fs::metadata(&out_lzw).unwrap().len();
        let none_size = std::fs::metadata(&out_none).unwrap().len();
        assert!(
            lzw_size < none_size,
            "LZW ({lzw_size}) should be smaller than none ({none_size})"
        );

        let back = decode_file(&out_lzw).unwrap();
        assert_eq!(back.width(), im.width());
        assert_eq!(back.height(), im.height());
        assert_eq!(back.data(), im.data(), "LZW TIFF must be lossless");
    }

    #[test]
    fn save_tiff_gray16_deflate_round_trips_via_decode_tiff_page() {
        let im = ramp_gray16(48, 32);
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("gray16.tif");
        im.save_tiff(&out, TiffCompression::Deflate).unwrap();

        let back = decode_tiff_page(&out, 0).unwrap();
        assert_eq!(back.width(), im.width());
        assert_eq!(back.height(), im.height());
        assert_eq!(back.format(), PixelFormat::Gray16);
        assert_eq!(
            back.data(),
            im.data(),
            "16-bit Deflate TIFF must be lossless"
        );
    }

    #[test]
    fn save_tiff_rgb8_deflate_round_trips_via_decode_tiff_page() {
        let im = ramp_rgb8(48, 32);
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("rgb8.tif");
        im.save_tiff(&out, TiffCompression::Deflate).unwrap();

        let back = decode_tiff_page(&out, 0).unwrap();
        assert_eq!(back.width(), im.width());
        assert_eq!(back.height(), im.height());
        assert_eq!(back.format(), PixelFormat::Rgb8);
        assert_eq!(
            back.data(),
            im.data(),
            "multi-channel 8-bit Deflate TIFF must be lossless"
        );
    }

    #[test]
    fn save_tiff_rgba16_deflate_round_trips_via_decode_tiff_page() {
        let im = ramp_rgba16(48, 32);
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("rgba16_deflate.tif");
        im.save_tiff(&out, TiffCompression::Deflate).unwrap();

        let back = decode_tiff_page(&out, 0).unwrap();
        assert_eq!(back.width(), im.width());
        assert_eq!(back.height(), im.height());
        assert_eq!(back.format(), PixelFormat::Rgba16);
        assert_eq!(
            back.data(),
            im.data(),
            "16-bit RGBA Deflate TIFF (predictor on) must be lossless"
        );
    }

    #[test]
    fn save_tiff_rgba16_lzw_round_trips_via_decode_tiff_page() {
        let im = ramp_rgba16(48, 32);
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("rgba16_lzw.tif");
        im.save_tiff(&out, TiffCompression::Lzw).unwrap();

        let back = decode_tiff_page(&out, 0).unwrap();
        assert_eq!(back.width(), im.width());
        assert_eq!(back.height(), im.height());
        assert_eq!(back.format(), PixelFormat::Rgba16);
        assert_eq!(
            back.data(),
            im.data(),
            "16-bit RGBA LZW TIFF (predictor on) must be lossless"
        );
    }

    #[test]
    fn tiff_save_tiff_load_round_trips_bit_exact() {
        let im = ramp_gray8(40, 40);
        let bytes = im.tiff_save();
        assert!(!bytes.is_empty(), "tiff_save must emit TIFF bytes");
        let back = Raster::tiff_load(&bytes).unwrap();
        assert_eq!(back.width(), im.width());
        assert_eq!(back.height(), im.height());
        assert_eq!(back.data(), im.data());
    }

    #[test]
    fn tiff_save_of_float_raster_yields_empty_and_tiff_load_errs() {
        // A float raster is a compute intermediate with no TIFF form, so the
        // infallible `tiff_save` yields an empty buffer (never a panic).
        let im = Raster::new(2, 2, PixelFormat::RgbaF32, vec![0u8; 2 * 2 * 16]).unwrap();
        let bytes = im.tiff_save();
        assert!(
            bytes.is_empty(),
            "float raster has no TIFF form and must yield an empty save"
        );
        // Loading the resulting empty buffer must surface a typed error, not
        // panic, so a silently-empty save is diagnosable downstream.
        assert!(Raster::tiff_load(&bytes).is_err());
        assert!(Raster::tiff_load(&[]).is_err());
    }

    #[test]
    fn tiff_page_count_and_decode_tiff_page_on_multipage() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multi.tif");
        std::fs::write(&path, multipage_gray8_fixture(3)).unwrap();

        assert_eq!(tiff_page_count(&path).unwrap(), 3);

        // The fixture writes page `p` (counting the IFD chain from zero) as
        // the constant `p * 10`, so the returned samples pin exactly which
        // image the index selected. `page` is a zero-based index over
        // `0..tiff_page_count`, matching `vips tiffload --page`, which loads
        // the first image at `--page 0` (measured against 8.18.4).
        for p in 0..3u32 {
            let page = decode_tiff_page(&path, p).unwrap();
            assert_eq!(page.width(), 4);
            assert_eq!(page.height(), 4);
            assert_eq!(
                page.data(),
                vec![(p as u8).saturating_mul(10); 16],
                "page {p} must be the {p}th image in the IFD chain"
            );
        }
    }

    #[test]
    fn decode_tiff_page_zero_is_the_first_image() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multi.tif");
        std::fs::write(&path, multipage_gray8_fixture(2)).unwrap();
        // Page 0 used to be rejected outright. It is now the first image,
        // written by the fixture as the constant 0.
        assert_eq!(decode_tiff_page(&path, 0).unwrap().data(), vec![0u8; 16]);
    }

    #[test]
    fn decode_tiff_page_rejects_a_page_past_the_end() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("multi.tif");
        std::fs::write(&path, multipage_gray8_fixture(3)).unwrap();

        // Valid indices for a three-page file are 0, 1 and 2, so 3 is the
        // first one past the end. vips draws the line in the same place:
        // `vips tiffload` on a single-page file with `--page 1` fails with
        // "TIFF does not contain page 1".
        assert!(decode_tiff_page(&path, 3).is_err());

        // The message has to carry both numbers, because "page 5" on its own
        // does not tell a caller whether they are one over or four over.
        let message = decode_tiff_page(&path, 5).unwrap_err().to_string();
        assert!(
            message.contains("page 5") && message.contains("has 3"),
            "out-of-range page must name the index and the count, got {message}"
        );
        assert!(
            message.contains("indexed from 0"),
            "out-of-range page must say which base it counted from, got {message}"
        );
    }

    #[test]
    fn decode_tiff_page_attaches_the_page_count() {
        let dir = tempfile::tempdir().unwrap();
        let multi = dir.path().join("multi.tif");
        std::fs::write(&multi, multipage_gray8_fixture(3)).unwrap();
        let single = dir.path().join("single.tif");
        std::fs::write(&single, multipage_gray8_fixture(1)).unwrap();

        // `n-pages` is the count and `page` is an index into `0..n-pages`, so
        // the count travels back with every page. vips's tiffload attaches the
        // same field to every TIFF it loads, single-page ones included:
        // `vipsheader -f n-pages` reports 3 for a three-page file and 1 for a
        // one-page file (measured against 8.18.4).
        for p in 0..3u32 {
            assert_eq!(
                decode_tiff_page(&multi, p).unwrap().get_n_pages(),
                3,
                "page {p} of a three-page file must report n-pages = 3"
            );
        }
        assert_eq!(decode_tiff_page(&single, 0).unwrap().get_n_pages(), 1);

        // Three pages and one page share a geometry, so the pair already
        // shows the count tracks the IFD chain rather than the picture. A
        // third point makes it a line, and five collides with nothing this
        // fixture carries: issue #635 is exactly the case where a plausible
        // number under this key turned out to be counting something else.
        let five = dir.path().join("five.tif");
        std::fs::write(&five, multipage_gray8_fixture(5)).unwrap();
        for p in 0..5u32 {
            assert_eq!(
                decode_tiff_page(&five, p).unwrap().get_n_pages(),
                5,
                "page {p} of a five-page file must report n-pages = 5"
            );
        }
    }

    /// Rewrite one tag's value in the first IFD of a little-endian classic
    /// TIFF, as a LONG/count-1 entry so any `u32` fits inline in the entry's
    /// own value slot. Used to make a fixture *declare* geometry it does not
    /// carry, which is the shape of a decompression bomb: the header lies and
    /// the decoder is asked to trust it.
    fn patch_first_ifd_tag(bytes: &mut [u8], tag: u16, value: u32) -> bool {
        assert_eq!(&bytes[0..2], b"II", "the tiff encoder writes little-endian");
        let ifd = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let entries = u16::from_le_bytes(bytes[ifd..ifd + 2].try_into().unwrap()) as usize;
        for i in 0..entries {
            let e = ifd + 2 + i * 12;
            if u16::from_le_bytes(bytes[e..e + 2].try_into().unwrap()) == tag {
                bytes[e + 2..e + 4].copy_from_slice(&4u16.to_le_bytes());
                bytes[e + 4..e + 8].copy_from_slice(&1u32.to_le_bytes());
                bytes[e + 8..e + 12].copy_from_slice(&value.to_le_bytes());
                return true;
            }
        }
        false
    }

    /// A fixture that declares `w` x `h` while carrying a 4x4 gray8 strip.
    /// `RowsPerStrip` moves with the height so the single strip the encoder
    /// wrote still satisfies the `tiff` crate's own strip-count consistency
    /// check, leaving the declared geometry as the only thing wrong with it.
    fn lying_geometry_fixture(w: u32, h: u32) -> Vec<u8> {
        let mut bytes = multipage_gray8_fixture(1);
        assert!(patch_first_ifd_tag(&mut bytes, 256, w), "ImageWidth");
        assert!(patch_first_ifd_tag(&mut bytes, 257, h), "ImageLength");
        assert!(patch_first_ifd_tag(&mut bytes, 278, h), "RowsPerStrip");
        bytes
    }

    /// Byte offset of the next-IFD pointer that closes each IFD in a
    /// little-endian classic TIFF, in chain order.
    fn ifd_next_pointer_offsets(bytes: &[u8]) -> Vec<usize> {
        assert_eq!(&bytes[0..2], b"II", "the tiff encoder writes little-endian");
        let mut offsets = Vec::new();
        let mut ifd = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        while ifd != 0 {
            let entries = u16::from_le_bytes(bytes[ifd..ifd + 2].try_into().unwrap()) as usize;
            let next = ifd + 2 + entries * 12;
            offsets.push(next);
            ifd = u32::from_le_bytes(bytes[next..next + 4].try_into().unwrap()) as usize;
        }
        offsets
    }

    /**
     * Tests that a TIFF whose header declares 100000x100000 is refused from
     * the declared geometry, before any pixel buffer is reserved. Until #540
     * this path never consulted `DecodeLimits` at all: it handed whatever came
     * back to `Raster::new`, whose `DEFAULT_MAX_ALLOC_BYTES` is 8 GiB, sixteen
     * times looser than the 512 MiB `DecodeLimits::max_alloc_bytes` the rest
     * of the crate publishes.
     * Works by patching the width, height and rows-per-strip tags of a real
     * 4x4 fixture so the file is structurally valid and only its declared size
     * is a lie, then asserting the typed variant with its field values.
     * Input: a 4x4 gray8 TIFF relabelled 100000x100000, default limits.
     */
    #[test]
    fn decode_tiff_page_rejects_declared_geometry_over_the_pixel_ceiling() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bomb.tif");
        std::fs::write(&path, lying_geometry_fixture(100_000, 100_000)).unwrap();
        assert!(matches!(
            decode_tiff_page(&path, 0),
            Err(DecodeError::DimensionLimitExceeded {
                width: 100_000,
                height: 100_000,
                max_pixels: 1_073_741_824
            })
        ));
    }

    /**
     * Tests the single-axis ceiling on the same path, which fires before the
     * pixel-count one and is the check the native `.v` reader and the
     * `image`-crate path already applied.
     * Works by declaring one axis past the default `max_coord` (10,000,000,
     * the libvips `VIPS_MAX_COORD`) while leaving the other small, so only
     * `check_coord` can catch it.
     * Input: a 4x4 gray8 TIFF relabelled 100000000x4, default limits.
     */
    #[test]
    fn decode_tiff_page_rejects_an_axis_over_the_coord_ceiling() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("wide.tif");
        std::fs::write(&path, lying_geometry_fixture(100_000_000, 4)).unwrap();
        assert!(matches!(
            decode_tiff_page(&path, 0),
            Err(DecodeError::CoordLimitExceeded {
                width: 100_000_000,
                height: 4,
                max_coord: 10_000_000
            })
        ));
    }

    /**
     * Tests the allocation budget, which neither of the other two ceilings
     * implies: a pixel count sees neither the band count nor the sample depth,
     * so the default 1-gigapixel `max_pixels` still waves through a 4 GiB
     * RGBA frame.
     * Works by choosing a ceiling above the compressed file on disk but below
     * the 4096-byte buffer a 64x64 gray8 page decodes into, so the file read
     * passes and only the pixel buffer trips; the test asserts the file really
     * is the smaller of the two rather than assuming it.
     * Input: a 64x64 gray8 Deflate TIFF with max_alloc_bytes just over the
     * file size.
     */
    #[test]
    fn decode_tiff_page_bounds_the_pixel_buffer() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("budget.tif");
        ramp_gray8(64, 64)
            .save_tiff(&path, TiffCompression::Deflate)
            .unwrap();
        let file_len = std::fs::metadata(&path).unwrap().len();
        assert!(
            file_len < 4096,
            "the deflate fixture must compress below its 4096-byte pixel buffer, got {file_len}"
        );
        let limits = DecodeLimits::default().with_max_alloc_bytes(file_len);
        assert!(matches!(
            decode_tiff_page_with_limits(&path, 0, limits),
            Err(DecodeError::AllocLimitExceeded {
                what: "TIFF page pixel buffer",
                needed_bytes: 4096,
                ..
            })
        ));
        // The same file decodes once the budget clears the buffer.
        let ok = DecodeLimits::default().with_max_alloc_bytes(4096);
        assert_eq!(
            decode_tiff_page_with_limits(&path, 0, ok).unwrap().data(),
            ramp_gray8(64, 64).data()
        );
    }

    /**
     * Tests that the whole-file read is bounded too. `decode_tiff_page` reads
     * the file into memory rather than streaming it, because the vips
     * multiband relabel patches the bytes before the decoder sees them, and a
     * plain `std::fs::read` sizes that buffer from the file with no ceiling at
     * all. That was the largest of the three unbounded reads #566 left on this
     * path.
     * Works by setting a ceiling below the file size and asserting the failure
     * names the file body rather than the pixel buffer, so it is clear the
     * read never happened.
     * Input: a 64x64 gray8 TIFF with max_alloc_bytes = 8.
     */
    #[test]
    fn decode_tiff_page_bounds_the_file_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("body.tif");
        ramp_gray8(64, 64)
            .save_tiff(&path, TiffCompression::Deflate)
            .unwrap();
        let file_len = std::fs::metadata(&path).unwrap().len();
        let limits = DecodeLimits::default().with_max_alloc_bytes(8);
        match decode_tiff_page_with_limits(&path, 0, limits) {
            Err(DecodeError::AllocLimitExceeded {
                what,
                needed_bytes,
                max_alloc_bytes,
            }) => {
                assert_eq!(what, "TIFF file body");
                assert_eq!(needed_bytes, file_len);
                assert_eq!(max_alloc_bytes, 8);
            }
            other => panic!("expected the file body to be refused, got {other:?}"),
        }
    }

    /**
     * Tests that the IFD walk stops at `DecodeLimits::max_pages` instead of
     * running to the end of the chain. #566 put this walk on the hot path:
     * `decode_tiff_page` calls it on every single page decode to source
     * `n-pages`, and nothing bounded it.
     * Works by pointing the last IFD's next-pointer at an offset past the end
     * of the file, so a walk that ran past the ceiling would fail with the
     * `tiff` crate's own malformed-chain error instead. Getting
     * `PageLimitExceeded` out of the low ceiling and a different error out of
     * the high one is the proof the low one never touched the bad link.
     * Input: a three-page 4x4 gray8 file with a fourth, unreadable link,
     * max_pages 2 then 100.
     */
    #[test]
    fn tiff_page_count_stops_at_the_page_ceiling() {
        let mut bytes = multipage_gray8_fixture(3);
        let nexts = ifd_next_pointer_offsets(&bytes);
        assert_eq!(nexts.len(), 3, "fixture must have three IFDs");
        let bogus = (bytes.len() as u32).saturating_add(4096);
        let last = *nexts.last().unwrap();
        bytes[last..last + 4].copy_from_slice(&bogus.to_le_bytes());

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("long.tif");
        std::fs::write(&path, &bytes).unwrap();

        let stopped = tiff_page_count_with_limits(&path, DecodeLimits::default().with_max_pages(2));
        assert!(
            matches!(
                stopped,
                Err(DecodeError::PageLimitExceeded { max_pages: 2 })
            ),
            "expected the ceiling to stop the walk, got {stopped:?}"
        );
        let walked =
            tiff_page_count_with_limits(&path, DecodeLimits::default().with_max_pages(100));
        assert!(
            !matches!(walked, Err(DecodeError::PageLimitExceeded { .. })),
            "a high ceiling must reach the broken link instead, got {walked:?}"
        );
        assert!(walked.is_err(), "the broken link must still be an error");
    }

    /**
     * Tests that the ceiling is inclusive, so a file with exactly `max_pages`
     * pages still reads. The bound has to be usable, not just safe: a
     * three-page file under a ceiling of three is not a bomb.
     * Input: a three-page 4x4 gray8 file, max_pages 3 then 2, through both the
     * count and the page decode.
     */
    #[test]
    fn the_page_ceiling_is_inclusive() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("three.tif");
        std::fs::write(&path, multipage_gray8_fixture(3)).unwrap();
        let at = DecodeLimits::default().with_max_pages(3);
        assert_eq!(tiff_page_count_with_limits(&path, at).unwrap(), 3);
        assert_eq!(
            decode_tiff_page_with_limits(&path, 2, at)
                .unwrap()
                .get_n_pages(),
            3
        );
        let under = DecodeLimits::default().with_max_pages(2);
        assert!(matches!(
            decode_tiff_page_with_limits(&path, 0, under),
            Err(DecodeError::PageLimitExceeded { max_pages: 2 })
        ));
    }

    /**
     * Tests that the in-memory round-trip honours the same ceilings. It shares
     * the decode path with the page readers, so leaving it on the old
     * `Raster::new` budget would have left half the hole open.
     * Works by handing `tiff_load_with_limits` a pixel ceiling of one, which
     * only the declared geometry can trip.
     * Input: a 4x4 gray8 TIFF in memory, max_pixels 1.
     */
    #[test]
    fn tiff_load_honours_the_decode_limits() {
        let bytes = multipage_gray8_fixture(1);
        let limits = DecodeLimits::default().with_max_pixels(1);
        assert!(matches!(
            Raster::tiff_load_with_limits(&bytes, limits),
            Err(DecodeError::DimensionLimitExceeded {
                width: 4,
                height: 4,
                max_pixels: 1
            })
        ));
        // The default form is unchanged for anything that fits.
        assert_eq!(Raster::tiff_load(&bytes).unwrap().width(), 4);
    }

    /**
     * Tests that the no-limits entry points really do delegate to the default
     * limits rather than keeping a second, looser path alive. Whether they do
     * is the entire subject of #540, and a delegation that silently stopped
     * delegating would be invisible from the outside otherwise.
     * Works by comparing each pair on a file that decodes, then on one that
     * the default ceilings refuse, and requiring the same outcome from both.
     * Input: a three-page fixture and a 100000x100000 liar.
     */
    #[test]
    fn the_default_entry_points_delegate_to_the_default_limits() {
        let dir = tempfile::tempdir().unwrap();
        let good = dir.path().join("good.tif");
        std::fs::write(&good, multipage_gray8_fixture(3)).unwrap();
        let d = DecodeLimits::default();
        assert_eq!(
            tiff_page_count(&good).unwrap(),
            tiff_page_count_with_limits(&good, d).unwrap()
        );
        assert_eq!(
            decode_tiff_page(&good, 1).unwrap().data(),
            decode_tiff_page_with_limits(&good, 1, d).unwrap().data()
        );

        let bad = dir.path().join("bad.tif");
        std::fs::write(&bad, lying_geometry_fixture(100_000, 100_000)).unwrap();
        assert_eq!(
            decode_tiff_page(&bad, 0).unwrap_err().to_string(),
            decode_tiff_page_with_limits(&bad, 0, d)
                .unwrap_err()
                .to_string()
        );
    }

    #[test]
    fn save_tiff_jpeg_is_unsupported_stub() {
        let im = ramp_gray8(8, 8);
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("jpeg.tif");
        let err = im.save_tiff(&out, TiffCompression::Jpeg).unwrap_err();
        assert!(matches!(err, SaveError::Encode(SinkError::Other(_))));
        assert!(err.to_string().to_lowercase().contains("not supported"));
    }

    #[test]
    fn save_tiff_ccitt_is_unsupported_stub() {
        let im = ramp_gray8(8, 8);
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("ccitt.tif");
        assert!(im.save_tiff(&out, TiffCompression::Ccitt).is_err());
    }

    #[test]
    fn save_tiff_jp2k_is_unsupported_stub() {
        let im = ramp_gray8(8, 8);
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("jp2k.tif");
        assert!(im.save_tiff(&out, TiffCompression::Jp2k).is_err());
    }

    #[test]
    fn save_bigtiff_is_unsupported_stub() {
        let im = ramp_gray8(8, 8);
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("big.tif");
        let err = im.save_bigtiff(&out, TiffCompression::None).unwrap_err();
        assert!(matches!(err, SaveError::Encode(SinkError::Other(_))));
    }

    #[test]
    fn save_tiff_tiled_is_unsupported_stub() {
        let im = ramp_gray8(8, 8);
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("tiled.tif");
        assert!(
            im.save_tiff_tiled(&out, TiffCompression::Jp2k, 128, 128, true, true)
                .is_err()
        );
    }

    /// Build an `n`-band uchar raster whose sample `(x, y, band)` is a
    /// distinct value, so a round-trip that scrambled bands or rows would be
    /// caught. `n` must not be 1/3/4 (those canonicalize to named formats).
    fn ramp_multi8(w: u32, h: u32, n: usize) -> Raster {
        let mut data = Vec::with_capacity(w as usize * h as usize * n);
        for y in 0..h {
            for x in 0..w {
                for b in 0..n {
                    data.push(((x as usize + y as usize * 7 + b * 40) & 0xFF) as u8);
                }
            }
        }
        let fmt = PixelFormat::with_channels(n, 1).unwrap();
        Raster::new(w, h, fmt, data).unwrap()
    }

    fn ramp_multi16(w: u32, h: u32, n: usize) -> Raster {
        let mut samples = Vec::with_capacity(w as usize * h as usize * n);
        for y in 0..h {
            for x in 0..w {
                for b in 0..n {
                    samples.push(((x as usize + y as usize * 100 + b * 5000) & 0xFFFF) as u16);
                }
            }
        }
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_ne_bytes()).collect();
        let fmt = PixelFormat::with_channels(n, 2).unwrap();
        Raster::new(w, h, fmt, bytes).unwrap()
    }

    /// A 5-band uchar multiband raster (vips writes it as RGB + 2 unassoc-alpha
    /// extra samples, which the `tiff` crate's `colortype()` rejects) must
    /// round-trip losslessly, back to `Multi8(5)` with identical samples.
    #[test]
    fn save_tiff_multiband8_round_trips_bit_exact() {
        let im = ramp_multi8(6, 4, 5);
        assert_eq!(im.format(), PixelFormat::with_channels(5, 1).unwrap());
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("multi5.tif");
        im.save_tiff(&out, TiffCompression::None).unwrap();

        let back = decode_tiff_page(&out, 0).unwrap();
        assert_eq!(back.width(), 6);
        assert_eq!(back.height(), 4);
        assert_eq!(back.format(), PixelFormat::with_channels(5, 1).unwrap());
        assert_eq!(back.data(), im.data(), "5-band uchar TIFF must be lossless");
    }

    /// A 2-band uchar raster (vips writes it as BlackIsZero + 1 unassoc-alpha
    /// extra sample, decoded by the `tiff` crate as `Multiband{num_samples:2}`)
    /// must round-trip back to `Multi8(2)`.
    #[test]
    fn save_tiff_multiband_two_band_round_trips() {
        let im = ramp_multi8(5, 3, 2);
        assert_eq!(im.format(), PixelFormat::with_channels(2, 1).unwrap());
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("multi2.tif");
        im.save_tiff(&out, TiffCompression::Deflate).unwrap();

        let back = decode_tiff_page(&out, 0).unwrap();
        assert_eq!(back.format(), PixelFormat::with_channels(2, 1).unwrap());
        assert_eq!(back.data(), im.data());
    }

    /// Build a vips-style RGB-photometric raster with `n - 3` unassociated-alpha
    /// extra samples: byte-for-byte the layout `vips tiffsave` emits for an
    /// `n >= 5`-band raster (confirmed with `tiffinfo`, which reports `RGB color`,
    /// `Extra Samples: 2<unassoc-alpha, ...>`, and `Samples/Pixel: n`). Pass a
    /// band count of five or more, so there is at least one extra sample beyond
    /// RGBA.
    fn rgb_plus_extra_multiband8(im: &Raster, n: usize) -> Vec<u8> {
        let w = im.width();
        let h = im.height();
        let wide = w * n as u32;
        let mut buf = Vec::new();
        {
            let mut encoder = TiffEncoder::new(Cursor::new(&mut buf)).unwrap();
            let mut image = encoder.new_image::<colortype::Gray8>(wide, h).unwrap();
            {
                let dir = image.encoder();
                dir.write_tag(Tag::ImageWidth, w).unwrap();
                dir.write_tag(Tag::SamplesPerPixel, n as u16).unwrap();
                dir.write_tag(Tag::BitsPerSample, &vec![8u16; n][..])
                    .unwrap();
                dir.write_tag(Tag::SampleFormat, &vec![1u16; n][..])
                    .unwrap();
                dir.write_tag(Tag::ExtraSamples, &vec![2u16; n - 3][..])
                    .unwrap();
                // The vips choice: RGB photometric with extra samples.
                dir.write_tag(Tag::PhotometricInterpretation, 2u16).unwrap();
            }
            image.write_data(im.data()).unwrap();
        }
        buf
    }

    /// A vips-native `>= 5`-band raster (RGB photometric + extra samples) decodes
    /// back to `Multi8(n)` with byte-identical samples. This also pins the
    /// load-bearing invariant that the `tiff` crate itself rejects RGB+extra
    /// (the reason [`encode_multiband`] writes `BlackIsZero`): if a crate upgrade
    /// started accepting it, the first assertion fails and the encode-layout
    /// choice can be revisited instead of silently drifting.
    #[test]
    fn decode_rgb_plus_extra_like_vips_round_trips() {
        let im = ramp_multi8(6, 4, 5);
        let bytes = rgb_plus_extra_multiband8(&im, 5);

        // Invariant pin: the raw `tiff` crate must still refuse RGB+extra.
        let mut raw = Decoder::new(Cursor::new(&bytes)).unwrap();
        assert!(
            raw.read_image().is_err(),
            "tiff crate unexpectedly accepts RGB+extra; revisit the encode layout"
        );

        // Our decode path relabels the photometric and recovers the exact
        // N-band samples vips stored.
        let back = Raster::tiff_load(&bytes).unwrap();
        assert_eq!(back.width(), 6);
        assert_eq!(back.height(), 4);
        assert_eq!(back.format(), PixelFormat::with_channels(5, 1).unwrap());
        assert_eq!(
            back.data(),
            im.data(),
            "vips RGB+extra 5-band TIFF must decode byte-exact"
        );

        // And a 3-band RGB / 4-band RGBA file (samples <= 4) is left untouched
        // by the relabel and still decodes through the native colour path.
        let rgb = ramp_rgb8(6, 4);
        let rgb_back = Raster::tiff_load(&rgb.tiff_save()).unwrap();
        assert_eq!(rgb_back.format(), PixelFormat::Rgb8);
        assert_eq!(rgb_back.data(), rgb.data());
    }

    /// A 6-band ushort raster round-trips losslessly through the in-memory
    /// `tiff_save` / `tiff_load` pair (Deflate), back to `Multi16(6)`.
    #[test]
    fn tiff_save_load_multiband16_round_trips() {
        let im = ramp_multi16(4, 5, 6);
        assert_eq!(im.format(), PixelFormat::with_channels(6, 2).unwrap());
        let bytes = im.tiff_save();
        assert!(!bytes.is_empty(), "multiband ushort must encode");
        let back = Raster::tiff_load(&bytes).unwrap();
        assert_eq!(back.width(), 4);
        assert_eq!(back.height(), 5);
        assert_eq!(back.format(), PixelFormat::with_channels(6, 2).unwrap());
        assert_eq!(back.data(), im.data());
    }

    /// The decoder reads the TIFF `Orientation` tag (274) that vips writes for
    /// an oriented raster, so `autorot` has a real cross-oracle. A file with
    /// tag 274 = 6 decodes to orientation 6; the absence of the tag reads as 1.
    #[test]
    fn decode_reads_tiff_orientation_tag() {
        // A plain Gray8 write has no Orientation tag → upright (1).
        let plain = ramp_gray8(4, 3);
        let plain_bytes = plain.tiff_save();
        assert_eq!(Raster::tiff_load(&plain_bytes).unwrap().orientation(), 1);

        // Hand-build a TIFF carrying Orientation = 6 (as vips does when it
        // saves an orientation-6 raster).
        let mut buf = Vec::new();
        {
            let mut encoder = TiffEncoder::new(Cursor::new(&mut buf)).unwrap();
            let mut image = encoder.new_image::<colortype::Gray8>(4, 3).unwrap();
            image.encoder().write_tag(Tag::Orientation, 6u16).unwrap();
            image.write_data(&[7u8; 12]).unwrap();
        }
        let back = Raster::tiff_load(&buf).unwrap();
        assert_eq!(back.orientation(), 6);
        // And autorot of a 4x3 orientation-6 raster rotates it to 3x4, as vips
        // (`vips autorot`) does.
        let rot = back.autorot();
        assert_eq!((rot.width(), rot.height()), (3, 4));
        assert_eq!(rot.orientation(), 1);
    }
}
