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
//! `n-pages` (readable through [`Raster::get_n_pages`]) is the matching
//! **count**, not an index, so the last page of a file is
//! `get_n_pages() - 1`. [`decode_tiff_page`] attaches it to every raster it
//! returns, single-page files included, which is what vips's `tiffload` does
//! (`vipsheader -f n-pages` reports 3 for a three-page file and 1 for a
//! one-page one).
//!
//! The crate's PDF readers ([`crate::extract_page_image`] and friends) are
//! **1-based**, and that is deliberate rather than an oversight left behind
//! here. A PDF carries its own page numbering, [`crate::PdfInfo`] reports that
//! numbering straight out of the document, and the CLI exposes `--page` to
//! users on those terms. The rule across the crate is that a document's own
//! page number is 1-based and a position in a sequence of frames is 0-based.
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
use crate::imageio::{MetadataValue, SaveError};
use crate::pixel::PixelFormat;
use crate::raster::Raster;
use crate::sink::SinkError;

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

/// Resolve the channel count of the current image.
///
/// `colortype()` maps the common photometrics and the `BlackIsZero`-with-extra
/// Multiband case. A vips-written `>= 5`-band raster is an RGB photometric with
/// extra samples, which `colortype()` rejects — but by the time the decoder
/// reaches here that file has already been relabelled to `BlackIsZero` by
/// [`normalize_multiband_photometric`], so it resolves as `Multiband{N}`. Any
/// residual colour type the decoder still refuses (e.g. CMYK-with-extra, which
/// vips does not write for these rasters) surfaces as a typed error rather than
/// a silent, wrong channel count.
fn resolve_channels<R: Read + Seek>(decoder: &mut Decoder<R>) -> Result<usize, DecodeError> {
    match decoder.colortype().map_err(tiff_decode_err)? {
        ColorType::Gray(_) => Ok(1),
        ColorType::RGB(_) => Ok(3),
        ColorType::RGBA(_) => Ok(4),
        ColorType::Multiband { num_samples, .. } => Ok(num_samples as usize),
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
    let little_endian = bytes.first() == Some(&b'I');
    let mut owned = bytes.to_vec();
    let value = if little_endian {
        1u16.to_le_bytes()
    } else {
        1u16.to_be_bytes()
    };
    for off in offsets {
        if let Some(slot) = owned.get_mut(off..off + 2) {
            slot.copy_from_slice(&value);
        }
    }
    Cow::Owned(owned)
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

/// Read the decoder's currently-selected image into a [`Raster`].
fn decode_current_image<R: Read + Seek>(decoder: &mut Decoder<R>) -> Result<Raster, DecodeError> {
    let (width, height) = decoder.dimensions().map_err(tiff_decode_err)?;
    let channels = resolve_channels(decoder)?;
    let orientation = read_tiff_orientation(decoder);
    let result = decoder.read_image().map_err(tiff_decode_err)?;
    let (format, data) = interpret(channels, result)?;
    let mut raster = Raster::new(width, height, format, data)?;
    raster.meta.orientation = orientation;
    Ok(raster)
}

/// Count the pages (IFDs) in a multi-page TIFF file.
///
/// # Errors
///
/// Returns [`DecodeError`] if the file cannot be opened or the TIFF header or
/// its IFD chain is malformed.
pub fn tiff_page_count(path: &Path) -> Result<u32, DecodeError> {
    let file = std::fs::File::open(path)?;
    let mut decoder = Decoder::new(std::io::BufReader::new(file)).map_err(tiff_decode_err)?;
    count_images(&mut decoder)
}

/// Walk a decoder's IFD chain to the end and report how many images it holds.
///
/// The decoder is left positioned on the last image, so a caller that also
/// wants pixels needs a second decoder over the same bytes.
fn count_images<R: Read + Seek>(decoder: &mut Decoder<R>) -> Result<u32, DecodeError> {
    let mut count = 1u32;
    while decoder.more_images() {
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
/// bound `page` has to stay under travels back with the pixels and is readable
/// through [`Raster::get_n_pages`]. `n-pages` is a count and `page` is an
/// index, which makes the last page of a file `get_n_pages() - 1`.
///
/// PDF page numbers in this crate are 1-based instead
/// ([`crate::extract_page_image`] and friends); see the module docs for why
/// the two conventions differ.
///
/// # Errors
///
/// Returns [`DecodeError`] if the file cannot be opened, `page` is at or past
/// the file's page count, the TIFF is malformed, or the page uses a color or
/// sample type this lane does not decode.
pub fn decode_tiff_page(path: &Path, page: u32) -> Result<Raster, DecodeError> {
    let bytes = std::fs::read(path)?;
    let bytes = normalize_multiband_photometric(&bytes);

    // Counting first buys two things: an out-of-range index reports the bound
    // it missed instead of whatever the `tiff` crate's seek happens to say,
    // and the count can ride back out on the raster as `n-pages`.
    let mut counter = Decoder::new(Cursor::new(bytes.as_ref())).map_err(tiff_decode_err)?;
    let n_pages = count_images(&mut counter)?;
    if page >= n_pages {
        return Err(decode_err(format!(
            "TIFF page {page} is out of range: pages are indexed from 0 and \
             this file has {n_pages}"
        )));
    }

    let mut decoder = Decoder::new(Cursor::new(bytes.as_ref())).map_err(tiff_decode_err)?;
    decoder
        .seek_to_image(page as usize)
        .map_err(tiff_decode_err)?;
    let mut raster = decode_current_image(&mut decoder)?;
    raster
        .fields
        .set("n-pages", MetadataValue::Int(i64::from(n_pages)));
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
    /// # Errors
    ///
    /// Returns [`DecodeError`] if the bytes are not a valid TIFF, or the
    /// first page uses a color or sample type this lane does not decode.
    pub fn tiff_load(data: &[u8]) -> Result<Raster, DecodeError> {
        let data = normalize_multiband_photometric(data);
        let mut decoder = Decoder::new(Cursor::new(data.as_ref())).map_err(tiff_decode_err)?;
        decode_current_image(&mut decoder)
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
