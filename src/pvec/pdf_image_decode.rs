//! Decode one PDF image into 8-bit samples (issue #1332): the filters, the colour spaces and
//! `/Decode`, an image mask, and the soft mask resampled into alpha.
//!
//! Private to [`super::pdf_image`]. It reads only what [`PdfImagePlacement`] recorded, so no
//! document is needed here.
//!
//! # The filter seam (issue #1362)
//!
//! A codec stage is [`FilterInput`] in, [`FilterOutput`] out, and [`decode_filter`] is the one
//! `match` that says which codec a name reaches. Each codec is a small module under
//! `pdf_filters/`, and the backend is picked at compile time by feature inside that module, so
//! there is no registry and no option to set.
//!
//! There is deliberately no `PdfFilterDecoder` trait. With one implementation per codec it
//! meets none of the five tests of the no-pass-through rule, and the types here are shaped so
//! that extracting it is one commit. The test that would justify it, written down so it can be
//! applied the day it passes: two backends for one filter in one build, or an outside consumer
//! injecting one.

use lopdf::{Dictionary, Object};

use super::pdf_image::{FilterError, PdfImageError, PdfImageNote};
use super::pdf_vector::{PdfImagePlacement, RawImage};
use crate::source::DecodeLimits;

/// What one filter stage gets: everything the dictionary says that a codec may need.
// Grouping: these fields vary together per image. Not every codec reads every field yet; the
// CCITT, JBIG2 and JPX issues (#1363 to #1365) are the readers.
#[allow(dead_code, reason = "read by the codecs that follow in #1363 to #1365")]
pub(super) struct FilterInput<'a> {
    /// `CCITTFaxDecode`, `JBIG2Decode`, `JPXDecode` or `DCTDecode`.
    pub(super) filter: &'a str,
    /// This stage's `/DecodeParms` entry, matched to the filter by position.
    pub(super) parms: Option<&'a Dictionary>,
    /// The stream after any `FlateDecode` stage.
    pub(super) data: &'a [u8],
    pub(super) width: u32,
    pub(super) height: u32,
    /// The declared `/BitsPerComponent`.
    pub(super) bpc: Option<u32>,
    /// The component count of the declared `/ColorSpace`, when there is one.
    pub(super) components: Option<usize>,
    /// The `/JBIG2Globals` stream content, for JBIG2 only.
    pub(super) globals: Option<&'a [u8]>,
    /// The one pixel budget, derived once from [`PdfImageOptions::max_pixels`].
    ///
    /// [`PdfImageOptions::max_pixels`]: super::pdf_image::PdfImageOptions::max_pixels
    pub(super) budget: DecodeLimits,
}

/// What a filter stage hands back.
pub(super) enum FilterOutput {
    /// Samples packed at `bpc`, rows padded to bytes, for the colour pipeline to interpret
    /// (`/ColorSpace`, `/Decode` and `/ImageMask` apply as for an unfiltered image).
    Packed { data: Vec<u8>, bpc: u32 },
    /// Eight-bit samples whose channel count the codec fixed. `space_from_codestream` says the
    /// codestream's colour space is the one to use whatever the dictionary says.
    Samples {
        data: Vec<u8>,
        channels: u8,
        space_from_codestream: bool,
        notes: Vec<PdfImageNote>,
    },
}

/// Whether `name` is a filter with a codec module.
fn is_codec(name: &str) -> bool {
    matches!(
        name,
        "DCTDecode" | "CCITTFaxDecode" | "JBIG2Decode" | "JPXDecode"
    )
}

/// One `match`, one place that knows which build has which codec.
fn decode_filter(input: FilterInput<'_>) -> Result<FilterOutput, PdfImageError> {
    use super::pdf_filters::{ccitt, dct, jbig2, jpx};
    match input.filter {
        "DCTDecode" => dct::decode(input),
        "CCITTFaxDecode" => ccitt::decode(input),
        "JBIG2Decode" => jbig2::decode(input),
        "JPXDecode" => jpx::decode(input),
        other => Err(PdfImageError::UnsupportedFilter(other.to_string())),
    }
}

/// The channels of decoded samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Channels {
    Gray,
    Rgb,
    Rgba,
}

impl Channels {
    pub(super) fn count(self) -> usize {
        match self {
            Self::Gray => 1,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }
}

/// An image as eight-bit samples.
pub(super) struct Decoded {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) channels: Channels,
    pub(super) data: Vec<u8>,
    pub(super) notes: Vec<PdfImageNote>,
    /// The source had one bit per component and the samples are gray: every sample is 0 or 255.
    pub(super) bilevel: bool,
    /// The last filter of the chain, or `none`: what `written_by_filter` is keyed by.
    pub(super) filter: String,
}

fn bad(detail: impl Into<String>) -> PdfImageError {
    PdfImageError::Decode(detail.into())
}

/// Decode `image`, refusing by name what the crate does not read.
pub(super) fn decode(image: &PdfImagePlacement, max_pixels: u64) -> Result<Decoded, PdfImageError> {
    if let Some((bytes, limit)) = image.stream_over_cap {
        return Err(PdfImageError::StreamTooLarge { bytes, limit });
    }
    let raw = image
        .raw
        .as_ref()
        .ok_or_else(|| bad("the image has no data"))?;
    // Applying the codestream's alpha is later work (tied to RGBA output and the gray-plus-alpha
    // question of #1329), so it is refused by name rather than dropped without a word.
    let smask_in_data = raw
        .dict
        .get(b"SMaskInData")
        .ok()
        .and_then(|o| o.as_i64().ok())
        .filter(|&n| n != 0);
    if let Some(n) = smask_in_data
        && filters(&raw.dict).last().is_some_and(|f| f == "JPXDecode")
    {
        return Err(PdfImageError::Filter {
            filter: "JPXDecode".to_string(),
            source: FilterError::SMaskInData(n),
        });
    }
    let (Some(width), Some(height)) = (image.width, image.height) else {
        return Err(bad("missing or invalid /Width or /Height"));
    };
    let pixels = u64::from(width) * u64::from(height);
    if pixels > max_pixels {
        return Err(PdfImageError::TooLarge {
            pixels,
            limit: max_pixels,
        });
    }
    let budget = DecodeLimits::default().with_max_pixels(max_pixels);
    let mut notes = Vec::new();
    let mut out = if image.image_mask {
        stencil(raw, width, height, image.fill_colour, budget)?
    } else {
        let space = match raw.dict.get(b"ColorSpace") {
            Ok(o) => Some(Space::parse(o)?),
            Err(_) => None,
        };
        let (samples, channels, one_bit) =
            colour_samples(raw, space, width, height, budget, &mut notes)?;
        Decoded {
            width,
            height,
            channels,
            data: samples,
            notes: Vec::new(),
            bilevel: one_bit && channels == Channels::Gray,
            filter: terminal_filter(raw),
        }
    };

    if let Ok(mask) = raw.dict.get(b"SMask") {
        let alpha = soft_mask(mask, width, height, budget)?;
        out = with_alpha(out, &alpha);
    }
    if raw.dict.has(b"Mask") {
        notes.push(PdfImageNote::MaskIgnored);
    }
    out.notes = notes;
    Ok(out)
}

/// The samples of `raw` as gray or RGB, whether the bake approximated anything, and whether the
/// source was packed at one bit per component.
fn colour_samples(
    raw: &RawImage,
    space: Option<Space>,
    width: u32,
    height: u32,
    budget: DecodeLimits,
    notes: &mut Vec<PdfImageNote>,
) -> Result<(Vec<u8>, Channels, bool), PdfImageError> {
    let bpc_declared = raw
        .dict
        .get(b"BitsPerComponent")
        .ok()
        .and_then(|o| o.as_i64().ok())
        .map(|v| u32::try_from(v).map_err(|_| bad("/BitsPerComponent is out of range")))
        .transpose()?;
    let filters = filters(&raw.dict);
    check_filters(&filters)?;
    let unfiltered = unfilter(
        raw,
        &filters,
        space.as_ref(),
        bpc_declared,
        width,
        height,
        budget,
    )?;
    let (data, bpc, ncomp, space, one_bit) = match unfiltered {
        FilterOutput::Packed { data, bpc } => {
            let space = space.ok_or_else(|| bad("the image has no /ColorSpace"))?;
            let n = space.components();
            (data, bpc, n, space, bpc == 1)
        }
        FilterOutput::Samples {
            data,
            channels,
            space_from_codestream,
            notes: codec_notes,
        } => {
            notes.extend(codec_notes);
            // The codec says how many components there are; a /ColorSpace that disagrees loses.
            let natural = match channels {
                1 => Space::Gray,
                4 => Space::Cmyk,
                _ => Space::Rgb,
            };
            let space = match space {
                Some(s) if !space_from_codestream && s.components() == usize::from(channels) => s,
                _ => natural,
            };
            (data, 8, usize::from(channels), space, false)
        }
    };
    if bpc == 16 {
        notes.push(PdfImageNote::BitDepthReduced);
    }
    if space.approximate() {
        notes.push(PdfImageNote::ColourSpaceApproximated);
    }
    // A JPEG 2000 codestream carries its own range: `/Decode` applies to it only through an
    // Indexed space.
    let decode = if filters.last().is_some_and(|f| f == "JPXDecode")
        && !matches!(space, Space::Indexed { .. })
    {
        space.default_decode(bpc)
    } else {
        decode_array(&raw.dict, &space, bpc, ncomp)?
    };
    let channels = if space.gray_output() {
        Channels::Gray
    } else {
        Channels::Rgb
    };
    let out = convert(&space, &decode, &data, width, height, ncomp, bpc, channels)?;
    Ok((out, channels, one_bit))
}

// ---- filters ----

fn filters(dict: &Dictionary) -> Vec<String> {
    match dict.get(b"Filter") {
        Ok(Object::Name(n)) => vec![String::from_utf8_lossy(n).into_owned()],
        Ok(Object::Array(a)) => a
            .iter()
            .filter_map(|o| o.as_name().ok())
            .map(|n| String::from_utf8_lossy(n).into_owned())
            .collect(),
        _ => Vec::new(),
    }
}

/// The chains the bake reads: nothing, `FlateDecode`, a codec, or `FlateDecode` then a codec.
fn check_filters(filters: &[String]) -> Result<(), PdfImageError> {
    let ok = match filters
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        [] | ["FlateDecode"] => true,
        [codec] | ["FlateDecode", codec] => is_codec(codec),
        _ => false,
    };
    if ok {
        return Ok(());
    }
    // Refuse by the first name that is not a plain Flate or a codec, so the report says which.
    let culprit = filters
        .iter()
        .find(|f| f.as_str() != "FlateDecode" && !is_codec(f))
        .or_else(|| filters.first())
        .cloned()
        .unwrap_or_default();
    Err(PdfImageError::UnsupportedFilter(culprit))
}

/// The last filter of the chain, or `none`.
fn terminal_filter(raw: &RawImage) -> String {
    filters(&raw.dict)
        .pop()
        .unwrap_or_else(|| "none".to_string())
}

/// The `/DecodeParms` entry for filter number `index`. An array is matched to the
/// filters by position, and a `null` or a missing entry is no parameters; a lone dictionary
/// belongs to the first filter.
fn parms_for(dict: &Dictionary, index: usize) -> Option<&Dictionary> {
    match dict.get(b"DecodeParms") {
        Ok(Object::Dictionary(d)) if index == 0 => Some(d),
        Ok(Object::Array(a)) => a.get(index).and_then(|o| o.as_dict().ok()),
        _ => None,
    }
}

fn unfilter(
    raw: &RawImage,
    filters: &[String],
    space: Option<&Space>,
    bpc: Option<u32>,
    width: u32,
    height: u32,
    budget: DecodeLimits,
) -> Result<FilterOutput, PdfImageError> {
    let ncomp = space.map_or(4, Space::components);
    let flate = filters.first().is_some_and(|f| f == "FlateDecode");
    let codec = filters.last().map(String::as_str).filter(|f| is_codec(f));
    let mut data = std::borrow::Cow::Borrowed(raw.content.as_slice());
    if flate {
        let cap = if codec.is_some() {
            // A codestream inside Flate has no row structure, so give it the room of a raw image.
            (width as usize)
                .saturating_mul(height as usize)
                .saturating_mul(4)
                .saturating_add(1 << 20)
        } else {
            let row_bytes = row_bytes(width, ncomp, bpc.unwrap_or(16));
            (row_bytes + 1)
                .saturating_mul(height as usize)
                .saturating_add(4096)
        };
        let mut inflated = crate::pdf::flate_decompress(&data, cap).map_err(PdfImageError::Pdf)?;
        if codec.is_none() {
            inflated = unpredict(parms_for(&raw.dict, 0), inflated)?;
        }
        data = std::borrow::Cow::Owned(inflated);
    }
    if let Some(filter) = codec {
        let index = filters.len() - 1;
        let globals = None;
        return decode_filter(FilterInput {
            filter,
            parms: parms_for(&raw.dict, index),
            data: &data,
            width,
            height,
            bpc,
            components: space.map(Space::components),
            globals,
            budget,
        });
    }
    let bpc = bpc.ok_or_else(|| bad("the image has no /BitsPerComponent"))?;
    Ok(FilterOutput::Packed {
        data: data.into_owned(),
        bpc,
    })
}

fn row_bytes(width: u32, ncomp: usize, bpc: u32) -> usize {
    (width as usize * ncomp * bpc as usize).div_ceil(8)
}

/// Undo `/Predictor` 2 (TIFF, 8 bits) and 10 to 15 (PNG) from `/DecodeParms`.
fn unpredict(parms: Option<&Dictionary>, data: Vec<u8>) -> Result<Vec<u8>, PdfImageError> {
    let get = |key: &[u8], default: i64| {
        parms
            .and_then(|d| d.get(key).ok())
            .and_then(|o| o.as_i64().ok())
            .unwrap_or(default)
    };
    let predictor = get(b"Predictor", 1);
    if predictor <= 1 {
        return Ok(data);
    }
    let colors = usize::try_from(get(b"Colors", 1)).unwrap_or(1).max(1);
    let pbpc = u32::try_from(get(b"BitsPerComponent", 8)).unwrap_or(8);
    let columns = usize::try_from(get(b"Columns", 1)).unwrap_or(1).max(1);
    let row = (columns * colors * pbpc as usize).div_ceil(8);
    let bpp = (colors * pbpc as usize).div_ceil(8).max(1);
    if row == 0 {
        return Err(bad("a predictor row of no bytes"));
    }
    if predictor == 2 {
        if pbpc != 8 {
            return Err(bad("a TIFF predictor is read at 8 bits per component only"));
        }
        let mut out = data;
        for line in out.chunks_mut(row) {
            for i in bpp..line.len() {
                line[i] = line[i].wrapping_add(line[i - bpp]);
            }
        }
        return Ok(out);
    }
    if !(10..=15).contains(&predictor) {
        return Err(bad(format!("unknown /Predictor {predictor}")));
    }
    let mut out = Vec::with_capacity(data.len());
    let mut prev = vec![0u8; row];
    for line in data.chunks(row + 1) {
        let (tag, body) = line
            .split_first()
            .ok_or_else(|| bad("empty predictor row"))?;
        let mut cur = body.to_vec();
        cur.resize(row, 0);
        for i in 0..row {
            let a = if i >= bpp { cur[i - bpp] } else { 0 };
            let b = prev[i];
            let c = if i >= bpp { prev[i - bpp] } else { 0 };
            let add = match tag {
                0 => 0,
                1 => a,
                2 => b,
                3 => ((u16::from(a) + u16::from(b)) / 2) as u8,
                4 => paeth(a, b, c),
                t => return Err(bad(format!("unknown PNG row filter {t}"))),
            };
            cur[i] = cur[i].wrapping_add(add);
        }
        out.extend_from_slice(&cur);
        prev = cur;
    }
    Ok(out)
}

fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let (ia, ib, ic) = (i16::from(a), i16::from(b), i16::from(c));
    let p = ia + ib - ic;
    let (pa, pb, pc) = ((p - ia).abs(), (p - ib).abs(), (p - ic).abs());
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

// ---- colour spaces ----

#[derive(Debug, Clone)]
enum Space {
    Gray,
    Rgb,
    Cmyk,
    CalGray {
        gamma: f64,
    },
    CalRgb {
        gamma: [f64; 3],
        matrix: [f64; 9],
        white: [f64; 3],
    },
    Lab {
        white: [f64; 3],
        range: [f64; 4],
    },
    /// An ICC profile, read as the device space with its component count.
    Icc(usize),
    Indexed {
        base: Box<Space>,
        hival: u32,
        lookup: Vec<u8>,
    },
}

fn numbers(o: Option<&Object>) -> Vec<f64> {
    o.and_then(|o| o.as_array().ok())
        .map(|a| {
            a.iter()
                .filter_map(|v| match v {
                    Object::Integer(i) => Some(*i as f64),
                    Object::Real(r) => Some(f64::from(*r)),
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default()
}

fn array_n<const N: usize>(v: &[f64], default: [f64; N]) -> [f64; N] {
    <[f64; N]>::try_from(v).unwrap_or(default)
}

impl Space {
    fn parse(o: &Object) -> Result<Self, PdfImageError> {
        match o {
            Object::Name(n) => match n.as_slice() {
                b"DeviceGray" | b"G" | b"CalGray" => Ok(Self::Gray),
                b"DeviceRGB" | b"RGB" | b"CalRGB" => Ok(Self::Rgb),
                b"DeviceCMYK" | b"CMYK" => Ok(Self::Cmyk),
                other => Err(PdfImageError::UnsupportedColourSpace(
                    String::from_utf8_lossy(other).into_owned(),
                )),
            },
            Object::Array(a) => {
                let family = a
                    .first()
                    .and_then(|f| f.as_name().ok())
                    .ok_or_else(|| bad("a /ColorSpace array without a family"))?;
                let dict = |i: usize| a.get(i).and_then(|o| o.as_dict().ok());
                match family {
                    b"ICCBased" => {
                        let n = match a.get(1) {
                            Some(Object::Stream(s)) => {
                                s.dict.get(b"N").ok().and_then(|n| n.as_i64().ok())
                            }
                            _ => None,
                        };
                        match n {
                            Some(1) => Ok(Self::Icc(1)),
                            Some(3) => Ok(Self::Icc(3)),
                            Some(4) => Ok(Self::Icc(4)),
                            _ => Err(bad("an /ICCBased space without /N 1, 3 or 4")),
                        }
                    }
                    b"CalGray" => {
                        let gamma = match dict(1).and_then(|d| d.get(b"Gamma").ok()) {
                            Some(Object::Integer(i)) => *i as f64,
                            Some(Object::Real(r)) => f64::from(*r),
                            _ => 1.0,
                        };
                        Ok(Self::CalGray { gamma })
                    }
                    b"CalRGB" => {
                        let d = dict(1);
                        let g = numbers(d.and_then(|d| d.get(b"Gamma").ok()));
                        let m = numbers(d.and_then(|d| d.get(b"Matrix").ok()));
                        let w = numbers(d.and_then(|d| d.get(b"WhitePoint").ok()));
                        Ok(Self::CalRgb {
                            gamma: array_n(&g, [1.0; 3]),
                            matrix: array_n(&m, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
                            white: array_n(&w, D65),
                        })
                    }
                    b"Lab" => {
                        let d = dict(1);
                        let w = numbers(d.and_then(|d| d.get(b"WhitePoint").ok()));
                        let r = numbers(d.and_then(|d| d.get(b"Range").ok()));
                        Ok(Self::Lab {
                            white: array_n(&w, D65),
                            range: array_n(&r, [-100.0, 100.0, -100.0, 100.0]),
                        })
                    }
                    b"Indexed" | b"I" => {
                        let base = a
                            .get(1)
                            .ok_or_else(|| bad("an /Indexed space without a base"))?;
                        let base = Self::parse(base)?;
                        if matches!(base, Self::Indexed { .. }) {
                            return Err(bad("an /Indexed space over an /Indexed space"));
                        }
                        let hival = a
                            .get(2)
                            .and_then(|o| o.as_i64().ok())
                            .and_then(|v| u32::try_from(v).ok())
                            .filter(|v| *v <= 255)
                            .ok_or_else(|| bad("an /Indexed hival outside 0..=255"))?;
                        let lookup = match a.get(3) {
                            Some(Object::String(s, _)) => s.clone(),
                            _ => return Err(bad("an /Indexed space without a lookup table")),
                        };
                        Ok(Self::Indexed {
                            base: Box::new(base),
                            hival,
                            lookup,
                        })
                    }
                    other => Err(PdfImageError::UnsupportedColourSpace(
                        String::from_utf8_lossy(other).into_owned(),
                    )),
                }
            }
            _ => Err(bad("a /ColorSpace that is neither a name nor an array")),
        }
    }

    fn components(&self) -> usize {
        match self {
            Self::Gray | Self::CalGray { .. } | Self::Indexed { .. } => 1,
            Self::Rgb | Self::CalRgb { .. } | Self::Lab { .. } => 3,
            Self::Cmyk => 4,
            Self::Icc(n) => *n,
        }
    }

    /// Whether the conversion to sRGB is an approximation the report names.
    fn approximate(&self) -> bool {
        match self {
            Self::Gray | Self::Rgb => false,
            Self::Indexed { base, .. } => base.approximate(),
            _ => true,
        }
    }

    fn gray_output(&self) -> bool {
        match self {
            Self::Gray | Self::CalGray { .. } | Self::Icc(1) => true,
            Self::Indexed { base, .. } => base.gray_output(),
            _ => false,
        }
    }

    /// Default `/Decode` for `bpc` bits.
    fn default_decode(&self, bpc: u32) -> Vec<f64> {
        match self {
            Self::Indexed { .. } => vec![0.0, f64::from((1u32 << bpc.min(16)) - 1)],
            Self::Lab { range, .. } => vec![0.0, 100.0, range[0], range[1], range[2], range[3]],
            other => (0..other.components()).flat_map(|_| [0.0, 1.0]).collect(),
        }
    }

    /// sRGB for components in the space's own domain.
    fn eval(&self, c: &[f64]) -> [u8; 3] {
        match self {
            Self::Gray | Self::Icc(1) => [unit8(c[0]); 3],
            Self::Rgb | Self::Icc(3) => [unit8(c[0]), unit8(c[1]), unit8(c[2])],
            // The existing PDF reader's formula: no black generation or colour management.
            Self::Cmyk | Self::Icc(_) => {
                let k = 1.0 - c[3].clamp(0.0, 1.0);
                let ch = |v: f64| unit8((1.0 - v.clamp(0.0, 1.0)) * k);
                [ch(c[0]), ch(c[1]), ch(c[2])]
            }
            Self::CalGray { gamma } => {
                let y = c[0].clamp(0.0, 1.0).powf(*gamma);
                [unit8(srgb_encode(y)); 3]
            }
            Self::CalRgb {
                gamma,
                matrix,
                white,
            } => {
                let l = [
                    c[0].clamp(0.0, 1.0).powf(gamma[0]),
                    c[1].clamp(0.0, 1.0).powf(gamma[1]),
                    c[2].clamp(0.0, 1.0).powf(gamma[2]),
                ];
                let xyz = [
                    matrix[0] * l[0] + matrix[3] * l[1] + matrix[6] * l[2],
                    matrix[1] * l[0] + matrix[4] * l[1] + matrix[7] * l[2],
                    matrix[2] * l[0] + matrix[5] * l[1] + matrix[8] * l[2],
                ];
                xyz_to_srgb(xyz, *white)
            }
            Self::Lab { white, range } => {
                let l = c[0].clamp(0.0, 100.0);
                let a = c[1].clamp(range[0], range[1]);
                let b = c[2].clamp(range[2], range[3]);
                let fy = (l + 16.0) / 116.0;
                let (fx, fz) = (fy + a / 500.0, fy - b / 200.0);
                let inv = |f: f64| {
                    if f * f * f > 0.008_856 {
                        f * f * f
                    } else {
                        (f - 16.0 / 116.0) / 7.787
                    }
                };
                xyz_to_srgb(
                    [white[0] * inv(fx), white[1] * inv(fy), white[2] * inv(fz)],
                    *white,
                )
            }
            Self::Indexed {
                base,
                hival,
                lookup,
            } => {
                let i = (c[0].round().max(0.0) as u32).min(*hival) as usize;
                let n = base.components();
                let d = base.default_decode(8);
                let comps: Vec<f64> = (0..n)
                    .map(|k| {
                        let v = f64::from(lookup.get(i * n + k).copied().unwrap_or(0));
                        d[2 * k] + v * (d[2 * k + 1] - d[2 * k]) / 255.0
                    })
                    .collect();
                base.eval(&comps)
            }
        }
    }
}

const D65: [f64; 3] = [0.9505, 1.0, 1.089];

fn unit8(v: f64) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5).floor() as u8
}

fn srgb_encode(v: f64) -> f64 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8 {
        12.92 * v
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// CIE XYZ relative to `white` to sRGB: scale to D65, then the sRGB matrix and its transfer.
fn xyz_to_srgb(xyz: [f64; 3], white: [f64; 3]) -> [u8; 3] {
    let x = xyz[0] * D65[0] / white[0].max(1e-6);
    let y = xyz[1] * D65[1] / white[1].max(1e-6);
    let z = xyz[2] * D65[2] / white[2].max(1e-6);
    let r = 3.2406 * x - 1.5372 * y - 0.4986 * z;
    let g = -0.9689 * x + 1.8758 * y + 0.0415 * z;
    let b = 0.0557 * x - 0.2040 * y + 1.0570 * z;
    [
        unit8(srgb_encode(r)),
        unit8(srgb_encode(g)),
        unit8(srgb_encode(b)),
    ]
}

fn decode_array(
    dict: &Dictionary,
    space: &Space,
    bpc: u32,
    ncomp: usize,
) -> Result<Vec<f64>, PdfImageError> {
    let given = numbers(dict.get(b"Decode").ok());
    if given.is_empty() {
        return Ok(space.default_decode(bpc));
    }
    if given.len() != 2 * ncomp {
        return Err(bad(format!(
            "/Decode has {} entries for {ncomp} components",
            given.len()
        )));
    }
    Ok(given)
}

// ---- samples to pixels ----

fn sample(row: &[u8], idx: usize, bpc: u32) -> u32 {
    match bpc {
        8 => u32::from(row[idx]),
        16 => u32::from(row[2 * idx]) << 8 | u32::from(row[2 * idx + 1]),
        1 | 2 | 4 => {
            let bit = idx * bpc as usize;
            let shift = 8 - bpc as usize - bit % 8;
            u32::from(row[bit / 8] >> shift) & ((1 << bpc) - 1)
        }
        _ => 0,
    }
}

#[allow(clippy::too_many_arguments)]
fn convert(
    space: &Space,
    decode: &[f64],
    data: &[u8],
    width: u32,
    height: u32,
    ncomp: usize,
    bpc: u32,
    channels: Channels,
) -> Result<Vec<u8>, PdfImageError> {
    if !matches!(bpc, 1 | 2 | 4 | 8 | 16) {
        return Err(bad(format!("{bpc} bits per component")));
    }
    let stride = row_bytes(width, ncomp, bpc);
    let expected = stride.saturating_mul(height as usize);
    if data.len() < expected {
        return Err(bad(format!(
            "the stream has {} bytes, the image needs {expected}",
            data.len()
        )));
    }
    let out_n = channels.count();
    let mut out = Vec::with_capacity(width as usize * height as usize * out_n);
    let default = decode == space.default_decode(bpc).as_slice();
    let device8 = bpc == 8
        && default
        && matches!(
            (space, ncomp),
            (Space::Gray | Space::Icc(1), 1) | (Space::Rgb | Space::Icc(3), 3)
        );
    for row in data.chunks(stride).take(height as usize) {
        if device8 {
            out.extend_from_slice(&row[..width as usize * ncomp]);
            continue;
        }
        let max = f64::from((1u32 << bpc) - 1);
        let mut comps = vec![0.0; ncomp];
        for x in 0..width as usize {
            for (k, slot) in comps.iter_mut().enumerate() {
                let v = f64::from(sample(row, x * ncomp + k, bpc));
                *slot = decode[2 * k] + v * (decode[2 * k + 1] - decode[2 * k]) / max;
            }
            let rgb = space.eval(&comps);
            match channels {
                Channels::Gray => out.push(rgb[0]),
                _ => out.extend_from_slice(&rgb),
            }
        }
    }
    Ok(out)
}

/// An image mask: 1 bit, painted in `fill` where the (decoded) sample is 0.
fn stencil(
    raw: &RawImage,
    width: u32,
    height: u32,
    fill: [u8; 3],
    budget: DecodeLimits,
) -> Result<Decoded, PdfImageError> {
    let filters = filters(&raw.dict);
    check_filters(&filters)?;
    if filters.iter().any(|f| f == "DCTDecode") {
        return Err(bad("an image mask cannot be a JPEG"));
    }
    let FilterOutput::Packed { data, .. } = unfilter(
        raw,
        &filters,
        Some(&Space::Gray),
        Some(1),
        width,
        height,
        budget,
    )?
    else {
        return Err(bad("an image mask cannot be a JPEG"));
    };
    let invert = numbers(raw.dict.get(b"Decode").ok()) == [1.0, 0.0];
    let stride = row_bytes(width, 1, 1);
    if data.len() < stride * height as usize {
        return Err(bad("the mask stream is shorter than the image"));
    }
    let mut out = Vec::with_capacity(width as usize * height as usize * 4);
    for row in data.chunks(stride).take(height as usize) {
        for x in 0..width as usize {
            let bit = sample(row, x, 1) == 1;
            let painted = bit == invert;
            out.extend_from_slice(&[fill[0], fill[1], fill[2], if painted { 255 } else { 0 }]);
        }
    }
    Ok(Decoded {
        width,
        height,
        channels: Channels::Rgba,
        data: out,
        notes: Vec::new(),
        bilevel: false,
        filter: terminal_filter(raw),
    })
}

/// The image's `/SMask` as one alpha byte per pixel of a `width` by `height` image.
fn soft_mask(
    mask: &Object,
    width: u32,
    height: u32,
    budget: DecodeLimits,
) -> Result<Vec<u8>, PdfImageError> {
    let max_pixels = budget.max_pixels;
    let Object::Stream(s) = mask else {
        return Err(bad("an /SMask that is not a stream"));
    };
    let dim = |key: &[u8]| {
        s.dict
            .get(key)
            .ok()
            .and_then(|o| o.as_i64().ok())
            .and_then(|v| u32::try_from(v).ok())
            .filter(|v| *v > 0)
            .ok_or_else(|| bad("a soft mask without /Width and /Height"))
    };
    let (mw, mh) = (dim(b"Width")?, dim(b"Height")?);
    let pixels = u64::from(mw) * u64::from(mh);
    if pixels > max_pixels {
        return Err(PdfImageError::TooLarge {
            pixels,
            limit: max_pixels,
        });
    }
    let raw = RawImage {
        dict: s.dict.clone(),
        content: s.content.clone(),
    };
    let space = match raw.dict.get(b"ColorSpace") {
        Ok(o) => Space::parse(o)?,
        Err(_) => Space::Gray,
    };
    let mut ignored = Vec::new();
    let (gray, channels, _) = colour_samples(&raw, Some(space), mw, mh, budget, &mut ignored)?;
    if channels != Channels::Gray {
        return Err(bad("a soft mask that is not a gray image"));
    }
    let mut out = Vec::with_capacity(width as usize * height as usize);
    for y in 0..height as usize {
        // Nearest sample by pixel centre.
        let sy = ((2 * y + 1) * mh as usize / (2 * height as usize)).min(mh as usize - 1);
        for x in 0..width as usize {
            let sx = ((2 * x + 1) * mw as usize / (2 * width as usize)).min(mw as usize - 1);
            out.push(gray[sy * mw as usize + sx]);
        }
    }
    Ok(out)
}

/// Add an alpha plane to gray or RGB samples (an RGBA image keeps its own alpha under the mask).
fn with_alpha(d: Decoded, alpha: &[u8]) -> Decoded {
    let n = d.channels.count();
    let mut data = Vec::with_capacity(d.width as usize * d.height as usize * 4);
    for (px, a) in d.data.chunks(n).zip(alpha) {
        match d.channels {
            Channels::Gray => data.extend_from_slice(&[px[0], px[0], px[0], *a]),
            Channels::Rgb => data.extend_from_slice(&[px[0], px[1], px[2], *a]),
            Channels::Rgba => data.extend_from_slice(&[px[0], px[1], px[2], px[3].min(*a)]),
        }
    }
    Decoded {
        channels: Channels::Rgba,
        data,
        ..d
    }
}
