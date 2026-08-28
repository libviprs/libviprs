//! AVIF still-image load: an AV1 keyframe in an ISOBMFF container.
//!
//! **This is not `heifload`, and it is not trying to be.** libvips reads
//! `.avif` through `heifload`, which is a libheif front end: libheif accepts
//! HEVC, AV1, AVC and JPEG payloads inside an HEIF container, and `heifsave`
//! writes **HEVC** by default. This module reads exactly one of those four
//! payloads and writes none of them, so naming it after the vips operation
//! would be a lie about three quarters of the surface. It is named for the
//! format it actually reads (issue #605, split out of #498, which closed
//! because HEVC has no pure-Rust decoder to port to).
//!
//! What that leaves out, precisely:
//!
//! * **No HEVC.** A `.heic` file, or an `.avif` whose primary item is `hvc1`,
//!   is refused with [`AvifError::UnsupportedCodec`]. This is the whole
//!   reason #498 closed and nothing here changes it.
//! * **No AVC and no JPEG payloads**, the other two item types libheif
//!   accepts, refused the same way.
//! * **No save side.** There is no pure-Rust AV1 encoder worth shipping in a
//!   pyramiding engine, and writing AVIF would reintroduce the parity problem
//!   that closed #498. `heifsave` has no counterpart here and is not deferred:
//!   it is out of scope.
//! * **No image sequences.** `avis` (the animated brand) is not read, and
//!   neither is any multi-item grid, overlay or tiled derivation. Still images
//!   only, which is what [`decode_avif`] says on the tin.
//!
//! # Operations
//!
//! | libviprs method | libvips equivalent | result |
//! |---|---|---|
//! | [`decode_avif`] | `heifload`, *for an AV1 still only* | 8-bit or 16-bit raster, 3 or 4 bands, [`Interpretation::Srgb`] / [`Interpretation::Rgb16`] |
//! | *(none)* | `heifsave` | deliberately absent, see above |
//!
//! # Why this is worth having anyway
//!
//! AV1 decoding is bit-exact by specification, so for the payloads this reads
//! the pixels match vips **exactly** rather than approximately. Very little
//! else in the foreign-format roadmap can say that: SVG text diverges
//! structurally, GIF palettes diverge by quantiser, JPEG bases diverge by
//! decoder. The tests in this module pin whole frames against
//! `oracle-captures/foreign-avif`, not tolerances.
//!
//! # Semantics
//!
//! * **Colour conversion is the risk, not the AV1 decode.** AV1 hands back
//!   YCbCr planes; libheif turns those into RGB, and *that* step is not
//!   fixed by any specification. Two conversions are implemented here and
//!   both are measured against the pinned oracle rather than reasoned about:
//!   the identity matrix (`matrix_coefficients = 0`), where the planes are
//!   already G, B, R and nothing is converted at all; and BT.601
//!   (`matrix_coefficients = 5` or `6`) at full range, where libheif's exact
//!   integer arithmetic is reproduced in [`ycbcr_to_rgb_420`]. Anything else,
//!   including BT.709 and limited-range video, is refused with
//!   [`AvifError::UnsupportedColour`] rather than converted by a formula
//!   nothing in this tree has checked. See that function for the numbers.
//! * **Chroma is upsampled nearest-neighbour**, because that is what libheif
//!   does and it is measured: a 4:2:0 plane pair is read at `(x >> 1, y >> 1)`.
//!   Measured over a 32x32 fixture, all 1024 pixels, and over the committed
//!   4x3 one; see `nearest_neighbour_chroma_matches_the_oracle`.
//! * **Deeper bit depths left-justify.** `heifload.c:1000-1016` does
//!   `((p[0] << 8) | p[1]) << (16 - bits_per_pixel)` on the samples libheif
//!   hands back, so a 10-bit sample comes back as `sample << 6` (low six bits
//!   always zero, maximum 65472) and a 12-bit one as `sample << 4` (maximum
//!   65520). libviprs matches, and the oracle's `bit_depth_carrier` record is
//!   where those two numbers come from.
//! * **A monochrome AVIF still returns three bands.** `heifload.c:763-765`
//!   decodes to RGB unconditionally ("FIXME .. we always decode to RGB in
//!   generate"), so a one-plane AV1 comes back with its luma repeated across
//!   R, G and B. Returning one band here would be a divergence, not a saving.
//! * **`ftyp` gates the file and does not pick the codec.** The oracle's
//!   `ftyp_brand_is_a_gate_not_a_codec_selector` record measured the same AV1
//!   payload behind twelve brands: all ten in libheif's magic list decode
//!   identically, and nine of them are *labelled* `heif-compression = hevc`
//!   while doing it, because vips echoes the brand rather than detecting the
//!   codec. So this module reads the codec off the item's `infe` type and its
//!   `av1C` property, never off the brand.
//! * **Sniffing is `ftyp` + major brand `avif`, and nothing wider.** The
//!   brand survey above shows vips reaching AV1 payloads behind `mif1`,
//!   `heic` and seven other brands. Claiming those here would mean claiming
//!   files whose payload is usually HEVC, which this cannot decode, so the
//!   sniffer stays on the one brand that names the format. A `mif1` file
//!   carrying AV1 is readable through [`decode_avif`] directly; it is simply
//!   not auto-detected.
//!
//! [`Interpretation::Srgb`]: crate::Interpretation::Srgb
//! [`Interpretation::Rgb16`]: crate::Interpretation::Rgb16

use crate::Interpretation;
use crate::pixel::PixelFormat;
use crate::raster::Raster;
use crate::source::{DecodeLimits, SourceError};

#[cfg(feature = "avif")]
use rav1d::include::dav1d::headers::Dav1dSequenceHeader;
#[cfg(feature = "avif")]
use rav1d::include::dav1d::picture::Dav1dPicture;

/// The signature [`crate::source::sniff`] matches an AVIF on: the `ftyp` box
/// type at offset 4 followed immediately by the major brand `avif`.
///
/// Eight bytes at offset 4, with **no constraint at offset 0**, because bytes
/// 0..4 are the `ftyp` box's own size and carry no signature. That is why the
/// route table needs [`crate::source::Magic::At`] and cannot express this as a
/// `Prefix` or a `Split`.
pub(crate) const MAGIC_AT_4: &[u8] = b"ftypavif";

/// Everything that can go wrong reading an AVIF.
#[derive(Debug, thiserror::Error)]
pub enum AvifError {
    /// The crate was built without the **`avif`** feature, so there is no AV1
    /// decoder behind [`decode_avif`] at all.
    ///
    /// Reported instead of a missing symbol or a panic, for the reason
    /// [`crate::jxl::JxlError::FeatureNotEnabled`] is: every entry point in
    /// this module exists at the same signature in both builds, so a caller's
    /// code does not change shape with the feature. It is the only variant a
    /// build without the feature produces and the one variant a build with it
    /// never produces, which is what separates "this build has no AV1
    /// decoder" from "these bytes are not AVIF" without reading a message.
    #[error("avif: AVIF decoding is not available in this build (enable the `avif` feature)")]
    FeatureNotEnabled,
    /// The ISOBMFF container is malformed: a box that runs past its parent, a
    /// required box missing, a length field that cannot be believed.
    #[error("avif: malformed container: {0}")]
    Container(&'static str),
    /// The primary item is not an AV1 image.
    ///
    /// This is the #498 wall, reported by name rather than as a generic
    /// parse failure: `hvc1` means the file needs an HEVC decoder, which is
    /// exactly what does not exist in pure Rust.
    #[error(
        "avif: primary item is `{found}`, not `av01`; this reads AV1 payloads only, not heifload"
    )]
    UnsupportedCodec {
        /// The four-character item type that was found instead.
        found: String,
    },
    /// The file declares a colour encoding this module has not measured
    /// against the oracle and will not guess at.
    ///
    /// See [`ycbcr_to_rgb_444`] for which two are implemented and why the rest
    /// are refused rather than approximated.
    #[error(
        "avif: unsupported colour encoding (matrix_coefficients={matrix}, full_range={full_range}); only identity and full-range BT.601 are measured"
    )]
    UnsupportedColour {
        /// The AV1 `matrix_coefficients` value.
        matrix: u8,
        /// The AV1 `color_range` flag: `true` is full range.
        full_range: bool,
    },
    /// The AV1 decoder refused the bitstream.
    #[error("avif: AV1 decode failed: {message}")]
    Decode {
        /// What the decoder said, or which call failed.
        message: String,
    },
    /// The decoded frame could not be wrapped in a [`Raster`].
    #[error("avif: {0}")]
    Raster(#[from] crate::raster::RasterError),
}

// ---------------------------------------------------------------------------
// ISOBMFF container
// ---------------------------------------------------------------------------
//
// Hand-rolled, and deliberately. The obvious crate for this is `avif-parse`,
// which is **MPL-2.0**; this project treated MPL as a blocker for #502 until
// resvg relicensed, so taking it here would quietly reverse a decision that
// was made on the record. The walk below is a few hundred lines and costs no
// lock entry at all.

/// A reader over a byte slice that cannot walk off the end.
///
/// Every multi-byte read is bounds-checked and returns
/// [`AvifError::Container`] rather than panicking, because every length in
/// this file is attacker-controlled. `truncated.avif` in the oracle fixtures
/// is the committed proof that a short file lands here and not in a panic.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], AvifError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or(AvifError::Container("length overflow"))?;
        let slice = self
            .bytes
            .get(self.pos..end)
            .ok_or(AvifError::Container("box runs past the end of its parent"))?;
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, AvifError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, AvifError> {
        let b = self.take(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32, AvifError> {
        let b = self.take(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64, AvifError> {
        let b = self.take(8)?;
        Ok(u64::from_be_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// An unsigned big-endian integer of `n` bytes, for the `iloc` fields
    /// whose width is declared in the box rather than fixed by the spec.
    fn uint(&mut self, n: usize) -> Result<u64, AvifError> {
        if n == 0 {
            return Ok(0);
        }
        if n > 8 {
            return Err(AvifError::Container("iloc field wider than 8 bytes"));
        }
        let mut v = 0u64;
        for &byte in self.take(n)? {
            v = (v << 8) | u64::from(byte);
        }
        Ok(v)
    }

    /// A `FullBox` version byte plus its three flag bytes.
    fn full_box(&mut self) -> Result<(u8, u32), AvifError> {
        let version = self.u8()?;
        let b = self.take(3)?;
        Ok((version, u32::from_be_bytes([0, b[0], b[1], b[2]])))
    }
}

/// One box header: its four-character type and the byte range of its payload
/// within the file.
struct BoxHeader {
    kind: [u8; 4],
    /// Absolute offsets into the whole file, so `iloc`'s file-relative
    /// offsets and a box payload can be talked about in the same units.
    body: std::ops::Range<usize>,
}

/// Walk the boxes directly inside `range`, calling `visit` for each.
///
/// Handles all three length encodings ISO/IEC 14496-12 allows: a 32-bit
/// `size`, `size == 1` meaning a 64-bit `largesize` follows the type, and
/// `size == 0` meaning "to the end of the enclosing box". A `size` between 1
/// and 7 is a malformed header rather than a zero-length box, and is refused:
/// accepting it would let a file loop forever on a box that never advances.
fn walk_boxes(
    bytes: &[u8],
    range: std::ops::Range<usize>,
    mut visit: impl FnMut(&BoxHeader, &[u8]) -> Result<(), AvifError>,
) -> Result<(), AvifError> {
    let end = range.end.min(bytes.len());
    let mut pos = range.start;
    while pos + 8 <= end {
        let mut r = Reader::new(&bytes[..end]);
        r.pos = pos;
        let size32 = r.u32()?;
        let kind: [u8; 4] = r.take(4)?.try_into().expect("4 bytes");
        let size = match size32 {
            0 => (end - pos) as u64,
            1 => r.u64()?,
            n if n < 8 => return Err(AvifError::Container("box size under the 8-byte header")),
            n => u64::from(n),
        };
        let size = usize::try_from(size).map_err(|_| AvifError::Container("box size overflow"))?;
        let box_end = pos
            .checked_add(size)
            .ok_or(AvifError::Container("box size overflow"))?;
        if box_end > end {
            return Err(AvifError::Container("box runs past the end of its parent"));
        }
        let header = BoxHeader {
            kind,
            body: r.pos..box_end,
        };
        visit(&header, bytes)?;
        // `size >= 8` is guaranteed above, so this always advances.
        pos = box_end;
    }
    Ok(())
}

/// The `av1C` decoder-configuration property (AV1-ISOBMFF section 2.3).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Av1Config {
    seq_profile: u8,
    high_bitdepth: bool,
    twelve_bit: bool,
    monochrome: bool,
    subsampling_x: bool,
    subsampling_y: bool,
}

impl Av1Config {
    /// Bits per sample this configuration declares: 8, 10 or 12.
    ///
    /// `twelve_bit` is only meaningful in `seq_profile` 2; in every other
    /// profile `high_bitdepth` alone selects 10-bit, which is why this is a
    /// nested test rather than two independent flags.
    fn bit_depth(self) -> u8 {
        match (self.seq_profile, self.high_bitdepth, self.twelve_bit) {
            (2, true, true) => 12,
            (_, true, _) => 10,
            _ => 8,
        }
    }
}

/// The `colr` box in its `nclx` form: the on-disk colour signalling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Nclx {
    matrix: u16,
    full_range: bool,
}

/// One entry of the `ipco` property container.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Property {
    Av1C(Av1Config),
    /// `ispe`, the declared display geometry. This is the geometry the
    /// allocation budget is priced against, and it is read **before** any
    /// AV1 decode, which is the point: an AVIF is a compressed container, so
    /// a 300-byte file can declare a 65535x65535 frame.
    Ispe {
        width: u32,
        height: u32,
    },
    Nclx(Nclx),
    /// `auxC`, whose URN is what marks an item as the alpha plane.
    AuxType(String),
    /// A property this module does not read. Kept as a placeholder so that
    /// `ipma`'s 1-based indices still line up.
    Other,
}

/// One item out of `iinf` / `iloc`.
#[derive(Clone, Debug)]
struct Item {
    id: u32,
    kind: [u8; 4],
    /// Absolute `(offset, length)` byte ranges in the file. More than one
    /// where an item is stored in several `mdat` extents.
    extents: Extents,
    /// 1-based indices into the `ipco` list.
    properties: Vec<u16>,
}

/// The parsed `meta` box: everything needed to find and describe the pixels.
struct Container {
    primary: u32,
    items: Vec<Item>,
    properties: Vec<Property>,
    /// `auxl` references: the aux item's id mapped to the item it describes.
    aux_of: Vec<(u32, u32)>,
}

impl Container {
    fn item(&self, id: u32) -> Option<&Item> {
        self.items.iter().find(|i| i.id == id)
    }

    /// The properties associated with `item`, in `ipma` order.
    fn properties_of(&self, item: &Item) -> impl Iterator<Item = &Property> {
        item.properties
            .iter()
            .filter_map(|&index| self.properties.get(usize::from(index).wrapping_sub(1)))
    }

    fn av1_config(&self, item: &Item) -> Option<Av1Config> {
        self.properties_of(item).find_map(|p| match p {
            Property::Av1C(c) => Some(*c),
            _ => None,
        })
    }

    fn geometry(&self, item: &Item) -> Option<(u32, u32)> {
        self.properties_of(item).find_map(|p| match p {
            Property::Ispe { width, height } => Some((*width, *height)),
            _ => None,
        })
    }

    fn nclx(&self, item: &Item) -> Option<Nclx> {
        self.properties_of(item).find_map(|p| match p {
            Property::Nclx(n) => Some(*n),
            _ => None,
        })
    }

    /// The alpha item for `master`, if the file carries one.
    ///
    /// Two conditions, both required: an `auxl` reference pointing at
    /// `master`, and an `auxC` property whose URN is the MPEG alpha one. The
    /// oracle's `alpha` record shows vips's own writer emitting exactly this
    /// shape, a second monochrome `av01` item linked by `auxl`.
    fn alpha_item(&self, master: u32) -> Option<&Item> {
        self.aux_of
            .iter()
            .filter(|&&(_, to)| to == master)
            .filter_map(|&(from, _)| self.item(from))
            .find(|item| {
                self.properties_of(item).any(|p| match p {
                    Property::AuxType(urn) => urn == ALPHA_URN,
                    _ => false,
                })
            })
    }
}

/// The `auxC` URN that marks an auxiliary item as the alpha plane
/// (ISO/IEC 23008-12). Measured in the oracle's `alpha` record, which shows
/// vips's own `heifsave` writing this exact string.
const ALPHA_URN: &str = "urn:mpeg:mpegB:cicp:systems:auxiliary:alpha";

/// Parse the `meta` box out of a whole AVIF file.
fn parse_container(bytes: &[u8]) -> Result<Container, AvifError> {
    let mut meta: Option<std::ops::Range<usize>> = None;
    let mut saw_ftyp = false;
    walk_boxes(bytes, 0..bytes.len(), |header, _| {
        match &header.kind {
            b"ftyp" => saw_ftyp = true,
            b"meta" if meta.is_none() => meta = Some(header.body.clone()),
            _ => {}
        }
        Ok(())
    })?;
    if !saw_ftyp {
        return Err(AvifError::Container("no ftyp box"));
    }
    let meta = meta.ok_or(AvifError::Container("no meta box"))?;

    // `meta` is a FullBox, so its children start four bytes into the body.
    let children = meta
        .start
        .checked_add(4)
        .ok_or(AvifError::Container("meta box overflow"))?..meta.end;
    if children.start > children.end {
        return Err(AvifError::Container("meta box shorter than its own header"));
    }

    let mut primary = None;
    let mut items: Vec<Item> = Vec::new();
    let mut locations: Vec<(u32, Extents)> = Vec::new();
    let mut properties = Vec::new();
    let mut associations: Vec<(u32, Vec<u16>)> = Vec::new();
    let mut aux_of = Vec::new();

    walk_boxes(bytes, children, |header, bytes| {
        match &header.kind {
            b"pitm" => primary = Some(parse_pitm(bytes, header.body.clone())?),
            b"iloc" => locations = parse_iloc(bytes, header.body.clone())?,
            b"iinf" => items = parse_iinf(bytes, header.body.clone())?,
            b"iref" => aux_of = parse_iref(bytes, header.body.clone())?,
            b"iprp" => {
                walk_boxes(bytes, header.body.clone(), |inner, bytes| {
                    match &inner.kind {
                        b"ipco" => properties = parse_ipco(bytes, inner.body.clone())?,
                        b"ipma" => associations = parse_ipma(bytes, inner.body.clone())?,
                        _ => {}
                    }
                    Ok(())
                })?;
            }
            _ => {}
        }
        Ok(())
    })?;

    let primary = primary.ok_or(AvifError::Container("no pitm box"))?;
    for item in &mut items {
        if let Some((_, extents)) = locations.iter().find(|(id, _)| *id == item.id) {
            item.extents.clone_from(extents);
        }
        if let Some((_, props)) = associations.iter().find(|(id, _)| *id == item.id) {
            item.properties.clone_from(props);
        }
    }
    Ok(Container {
        primary,
        items,
        properties,
        aux_of,
    })
}

/// `pitm`: which item is the image the file is *of*.
fn parse_pitm(bytes: &[u8], body: std::ops::Range<usize>) -> Result<u32, AvifError> {
    let mut r = Reader::new(&bytes[..body.end]);
    r.pos = body.start;
    let (version, _) = r.full_box()?;
    if version == 0 {
        Ok(u32::from(r.u16()?))
    } else {
        r.u32()
    }
}

/// `iloc`: where each item's bytes live.
///
/// Only `construction_method` 0 (file offsets) is accepted. Method 1 stores
/// the item inside an `idat` box and method 2 inside another item; neither is
/// something vips's own writer emits and neither is measured here, so both
/// are refused rather than silently read from the wrong place.
fn parse_iloc(
    bytes: &[u8],
    body: std::ops::Range<usize>,
) -> Result<Vec<(u32, Extents)>, AvifError> {
    let mut r = Reader::new(&bytes[..body.end]);
    r.pos = body.start;
    let (version, _) = r.full_box()?;
    let sizes = r.u8()?;
    let (offset_size, length_size) = (usize::from(sizes >> 4), usize::from(sizes & 0x0f));
    let sizes = r.u8()?;
    let (base_size, index_size) = (usize::from(sizes >> 4), usize::from(sizes & 0x0f));
    // The index field only exists in versions 1 and 2.
    let index_size = if version >= 1 { index_size } else { 0 };
    let count = if version < 2 {
        u32::from(r.u16()?)
    } else {
        r.u32()?
    };

    let mut out = Vec::new();
    for _ in 0..count {
        let id = if version < 2 {
            u32::from(r.u16()?)
        } else {
            r.u32()?
        };
        if version >= 1 {
            let method = r.u16()? & 0x0f;
            if method != 0 {
                return Err(AvifError::Container(
                    "iloc construction_method other than file offset",
                ));
            }
        }
        let _data_reference_index = r.u16()?;
        let base = r.uint(base_size)?;
        let extent_count = r.u16()?;
        let mut extents = Vec::new();
        for _ in 0..extent_count {
            if index_size > 0 {
                let _ = r.uint(index_size)?;
            }
            let offset = r.uint(offset_size)?;
            let length = r.uint(length_size)?;
            let start = base
                .checked_add(offset)
                .and_then(|v| usize::try_from(v).ok())
                .ok_or(AvifError::Container("iloc extent offset overflow"))?;
            let length = usize::try_from(length)
                .map_err(|_| AvifError::Container("iloc extent too long"))?;
            extents.push((start, length));
        }
        out.push((id, extents));
    }
    Ok(out)
}

/// `iinf` and its `infe` children: what each item *is*.
fn parse_iinf(bytes: &[u8], body: std::ops::Range<usize>) -> Result<Vec<Item>, AvifError> {
    let mut r = Reader::new(&bytes[..body.end]);
    r.pos = body.start;
    let (version, _) = r.full_box()?;
    // The entry count is 16-bit in version 0 and 32-bit from version 1.
    if version == 0 {
        let _ = r.u16()?;
    } else {
        let _ = r.u32()?;
    }
    let mut items = Vec::new();
    walk_boxes(bytes, r.pos..body.end, |header, bytes| {
        if &header.kind == b"infe" {
            items.push(parse_infe(bytes, header.body.clone())?);
        }
        Ok(())
    })?;
    Ok(items)
}

/// An item's stored byte ranges: absolute `(offset, length)` pairs.
type Extents = Vec<(usize, usize)>;

/// One `infe` entry: an item id and its four-character type.
///
/// Versions 0 and 1 carry no `item_type` at all (they predate the generic
/// item model and are always image items); versions 2 and 3 carry one. A
/// version-0 or -1 entry is reported with an all-zero type, which the codec
/// check below then refuses by name rather than mistaking for `av01`.
fn parse_infe(bytes: &[u8], body: std::ops::Range<usize>) -> Result<Item, AvifError> {
    let mut r = Reader::new(&bytes[..body.end]);
    r.pos = body.start;
    let (version, _) = r.full_box()?;
    let id = match version {
        0 | 1 => {
            let id = u32::from(r.u16()?);
            return Ok(Item {
                id,
                kind: [0; 4],
                extents: Vec::new(),
                properties: Vec::new(),
            });
        }
        2 => u32::from(r.u16()?),
        _ => r.u32()?,
    };
    let _protection_index = r.u16()?;
    let kind: [u8; 4] = r.take(4)?.try_into().expect("4 bytes");
    Ok(Item {
        id,
        kind,
        extents: Vec::new(),
        properties: Vec::new(),
    })
}

/// `iref`: the `auxl` references that link an alpha plane to its image.
fn parse_iref(bytes: &[u8], body: std::ops::Range<usize>) -> Result<Vec<(u32, u32)>, AvifError> {
    let mut r = Reader::new(&bytes[..body.end]);
    r.pos = body.start;
    let (version, _) = r.full_box()?;
    let wide = version >= 1;
    let mut out = Vec::new();
    walk_boxes(bytes, r.pos..body.end, |header, bytes| {
        if &header.kind != b"auxl" {
            return Ok(());
        }
        let mut r = Reader::new(&bytes[..header.body.end]);
        r.pos = header.body.start;
        let from = if wide { r.u32()? } else { u32::from(r.u16()?) };
        let count = r.u16()?;
        for _ in 0..count {
            let to = if wide { r.u32()? } else { u32::from(r.u16()?) };
            out.push((from, to));
        }
        Ok(())
    })?;
    Ok(out)
}

/// `ipco`: the property container, in the order `ipma` indexes it.
///
/// Every child produces exactly one entry, including the ones this module
/// does not read, because `ipma` refers to properties by **1-based position**
/// in this list. Skipping an unknown box would shift every index after it.
fn parse_ipco(bytes: &[u8], body: std::ops::Range<usize>) -> Result<Vec<Property>, AvifError> {
    let mut out = Vec::new();
    walk_boxes(bytes, body, |header, bytes| {
        let property = match &header.kind {
            b"av1C" => parse_av1c(bytes, header.body.clone())?,
            b"ispe" => parse_ispe(bytes, header.body.clone())?,
            b"colr" => parse_colr(bytes, header.body.clone())?,
            b"auxC" => parse_auxc(bytes, header.body.clone())?,
            _ => Property::Other,
        };
        out.push(property);
        Ok(())
    })?;
    Ok(out)
}

/// `av1C`: the AV1 codec configuration record.
fn parse_av1c(bytes: &[u8], body: std::ops::Range<usize>) -> Result<Property, AvifError> {
    let mut r = Reader::new(&bytes[..body.end]);
    r.pos = body.start;
    let marker_version = r.u8()?;
    if marker_version != 0x81 {
        return Err(AvifError::Container("av1C marker/version is not 1"));
    }
    let profile_level = r.u8()?;
    let flags = r.u8()?;
    Ok(Property::Av1C(Av1Config {
        seq_profile: profile_level >> 5,
        high_bitdepth: flags & 0b0100_0000 != 0,
        twelve_bit: flags & 0b0010_0000 != 0,
        monochrome: flags & 0b0001_0000 != 0,
        subsampling_x: flags & 0b0000_1000 != 0,
        subsampling_y: flags & 0b0000_0100 != 0,
    }))
}

/// `ispe`: the declared display geometry, and the number the allocation
/// budget is priced against.
fn parse_ispe(bytes: &[u8], body: std::ops::Range<usize>) -> Result<Property, AvifError> {
    let mut r = Reader::new(&bytes[..body.end]);
    r.pos = body.start;
    let (_version, _) = r.full_box()?;
    Ok(Property::Ispe {
        width: r.u32()?,
        height: r.u32()?,
    })
}

/// `colr`: colour signalling, in its `nclx` form only.
///
/// `rICC` and `prof` carry an ICC profile rather than a matrix, and neither
/// changes how the YCbCr planes are converted, so both land on
/// [`Property::Other`] and leave the matrix to the AV1 sequence header.
fn parse_colr(bytes: &[u8], body: std::ops::Range<usize>) -> Result<Property, AvifError> {
    let mut r = Reader::new(&bytes[..body.end]);
    r.pos = body.start;
    let kind = r.take(4)?;
    if kind != b"nclx" {
        return Ok(Property::Other);
    }
    let _primaries = r.u16()?;
    let _transfer = r.u16()?;
    let matrix = r.u16()?;
    let full_range = r.u8()? & 0x80 != 0;
    Ok(Property::Nclx(Nclx { matrix, full_range }))
}

/// `auxC`: the auxiliary type URN, which is how an alpha plane says so.
fn parse_auxc(bytes: &[u8], body: std::ops::Range<usize>) -> Result<Property, AvifError> {
    let mut r = Reader::new(&bytes[..body.end]);
    r.pos = body.start;
    let (_version, _) = r.full_box()?;
    let rest = r.take(r.remaining())?;
    let urn = rest.split(|&b| b == 0).next().unwrap_or(rest);
    Ok(Property::AuxType(String::from_utf8_lossy(urn).into_owned()))
}

/// `ipma`: which properties belong to which item.
fn parse_ipma(
    bytes: &[u8],
    body: std::ops::Range<usize>,
) -> Result<Vec<(u32, Vec<u16>)>, AvifError> {
    let mut r = Reader::new(&bytes[..body.end]);
    r.pos = body.start;
    let (version, flags) = r.full_box()?;
    // Bit 0 of the flags widens each property index from 7 bits to 15.
    let wide_index = flags & 1 != 0;
    let count = r.u32()?;
    let mut out = Vec::new();
    for _ in 0..count {
        let id = if version < 1 {
            u32::from(r.u16()?)
        } else {
            r.u32()?
        };
        let association_count = r.u8()?;
        let mut props = Vec::new();
        for _ in 0..association_count {
            // The top bit is `essential`, which this module does not act on:
            // it marks a property a reader must understand to render the
            // item, and every property here is either understood or ignored
            // by name rather than by that flag.
            let index = if wide_index {
                r.u16()? & 0x7fff
            } else {
                u16::from(r.u8()? & 0x7f)
            };
            props.push(index);
        }
        out.push((id, props));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Colour
// ---------------------------------------------------------------------------

/// AV1 `matrix_coefficients` 0: the identity matrix.
///
/// The "YCbCr" planes are really G, B and R in that order, so there is no
/// conversion to do and nothing to round. This is what `heifsave --lossless`
/// writes, and it is why the lossless fixtures round-trip byte for byte.
const MATRIX_IDENTITY: u16 = 0;
/// AV1 `matrix_coefficients` 5 and 6: BT.470-B/G and BT.601, which share the
/// same luma coefficients and therefore the same conversion.
const MATRIX_BT601_625: u16 = 5;
/// See [`MATRIX_BT601_625`].
const MATRIX_BT601_525: u16 = 6;
/// AV1 `matrix_coefficients` 2: unspecified.
///
/// libheif treats an unspecified matrix as BT.601, which is what the
/// `rgb8_icc` fixture exercises: it carries no `colr` box at all, so the
/// matrix comes from the AV1 sequence header, and vips still returns the
/// BT.601 answer.
const MATRIX_UNSPECIFIED: u16 = 2;

/// libheif's YCbCr to RGB conversion for a **4:4:4** frame.
///
/// The constants are the textbook full-range BT.601 ones and the arithmetic
/// is floating point with round-to-nearest. Measured over every one of the
/// 1024 pixels of a 32x32 4:4:4 fixture: **0 mismatches**. Truncating instead
/// of rounding gets 883 of those 1024 wrong, which is the positive control
/// for the rounding term.
///
/// No fixture in the tree contains an exact `.5` result, so round-half-up and
/// round-half-even are indistinguishable here and this picks Rust's
/// `f64::round` (half away from zero) without claiming measurement for that
/// last bit.
///
/// **This is not the 4:2:0 formula**, and that is the surprising part. See
/// [`ycbcr_to_rgb_420`]: libheif reaches the two chroma layouts through
/// different conversion ops and they do not agree to the last bit, so a
/// single shared implementation cannot match vips on both. Using this one on
/// 4:2:0 data gets 124 of 1024 pixels wrong; using that one here gets 103.
#[must_use]
pub fn ycbcr_to_rgb_444(y: u8, cb: u8, cr: u8) -> [u8; 3] {
    let luma = f64::from(y);
    let cb = f64::from(cb) - 128.0;
    let cr = f64::from(cr) - 128.0;
    let clamp = |v: f64| -> u8 { v.round().clamp(0.0, 255.0) as u8 };
    [
        clamp(luma + 1.402 * cr),
        clamp(luma - 0.344_136 * cb - 0.714_136 * cr),
        clamp(luma + 1.772 * cb),
    ]
}

/// libheif's YCbCr to RGB conversion for a **4:2:0** frame.
///
/// Here the arithmetic is 8.8 fixed point with a rounding term, not floating
/// point:
///
/// ```text
/// base = 256 * y + 128
/// r    = (base            + 359 * cr) >> 8
/// g    = (base -  88 * cb - 183 * cr) >> 8
/// b    = (base + 454 * cb           ) >> 8
/// ```
///
/// Measured over every one of the 1024 pixels of a 32x32 4:2:0 fixture:
/// **0 mismatches**. The float form in [`ycbcr_to_rgb_444`] gets 124 of them
/// wrong, every one by exactly one, and it also misses a single blue channel
/// of the committed 4x3 `rgb8_q50_420.avif`. Dropping the `+ 128` rounding
/// term so the shift truncates moves the errors the other way, which is that
/// term's positive control.
///
/// That two chroma layouts need two different conversions is not a guess
/// about libheif's source; it is what the numbers say, in both directions,
/// over 2048 measured pixels.
#[must_use]
pub fn ycbcr_to_rgb_420(y: u8, cb: u8, cr: u8) -> [u8; 3] {
    let base = 256 * i32::from(y) + 128;
    let cb = i32::from(cb) - 128;
    let cr = i32::from(cr) - 128;
    let clamp = |v: i32| -> u8 { v.clamp(0, 255) as u8 };
    [
        clamp((base + 359 * cr) >> 8),
        clamp((base - 88 * cb - 183 * cr) >> 8),
        clamp((base + 454 * cb) >> 8),
    ]
}

/// Left-justify a `bit_depth`-bit sample into the top of a `u16`.
///
/// `heifload.c:1000-1016` does `((p[0] << 8) | p[1]) << (16 - bits_per_pixel)`
/// on the big-endian samples libheif hands back, so a 10-bit sample becomes
/// `sample << 6` and a 12-bit one `sample << 4`. The consequence a caller
/// notices is that the low bits are always zero and the maximum is 65472 at
/// 10 bits and 65520 at 12, which is what the oracle's `bit_depth_carrier`
/// record measured and what `deep_samples_left_justify` pins.
#[must_use]
fn left_justify(sample: u16, bit_depth: u8) -> u16 {
    sample << (16u8.saturating_sub(bit_depth))
}

// ---------------------------------------------------------------------------
// Decode
// ---------------------------------------------------------------------------

/// One decoded AV1 frame, in sample units at the frame's own bit depth.
struct Frame {
    width: usize,
    height: usize,
    chroma_width: usize,
    chroma_height: usize,
    subsampling_x: u8,
    subsampling_y: u8,
    bit_depth: u8,
    monochrome: bool,
    matrix: u16,
    full_range: bool,
    /// Y, Cb, Cr. The chroma planes are empty for a monochrome frame.
    planes: [Vec<u16>; 3],
}

impl Frame {
    fn luma(&self, x: usize, y: usize) -> u16 {
        self.planes[0][y * self.width + x]
    }

    /// The chroma sample covering `(x, y)`, upsampled nearest-neighbour.
    ///
    /// Nearest rather than bilinear because that is what libheif does and it
    /// is measured: see `nearest_neighbour_chroma_matches_the_oracle`. The
    /// `min` guards the last column and row of an odd-sized image, where
    /// `x >> 1` can reach the plane width; `odd3x3_q50.avif` is the committed
    /// 3x3 fixture that exercises exactly that.
    fn chroma(&self, plane: usize, x: usize, y: usize) -> u16 {
        let cx = (x >> self.subsampling_x).min(self.chroma_width.saturating_sub(1));
        let cy = (y >> self.subsampling_y).min(self.chroma_height.saturating_sub(1));
        self.planes[plane][cy * self.chroma_width + cx]
    }
}

/// Decode an AVIF still image.
///
/// Reads the ISOBMFF container, finds the primary item, decodes its AV1
/// keyframe and converts the result to RGB or RGBA. An alpha plane stored as
/// an `auxl`-linked auxiliary item is decoded too and becomes the fourth
/// band.
///
/// **This is not `heifload`.** It reads AV1 payloads only: no HEVC, no AVC,
/// no JPEG-in-HEIF, no image sequences and no save side. See the module
/// documentation for the full list of what that excludes and why.
///
/// # Errors
///
/// * [`AvifError::FeatureNotEnabled`] when the crate was built without the
///   **`avif`** feature. Everything before the AV1 decode still runs, so a
///   malformed container and an over-budget frame are reported the same way
///   in both builds.
/// * [`AvifError::Container`] when the ISOBMFF structure is malformed.
/// * [`AvifError::UnsupportedCodec`] when the primary item is not `av01`,
///   which is what an HEVC `.heic` lands on.
/// * [`AvifError::UnsupportedColour`] for a colour encoding this module has
///   not measured; see [`ycbcr_to_rgb_444`].
/// * [`SourceError::CoordLimitExceeded`] when either declared axis exceeds
///   [`DecodeLimits::max_coord`].
/// * [`SourceError::DimensionLimitExceeded`] when `width * height` exceeds
///   [`DecodeLimits::max_pixels`].
/// * [`SourceError::AllocLimitExceeded`] when the frame the container
///   declares costs more than [`DecodeLimits::max_alloc_bytes`]. Checked
///   against `ispe` **before** the AV1 decode runs, which is the point: AVIF
///   is a compressed container, so a 300-byte file can declare a
///   65535x65535 frame.
pub fn decode_avif(bytes: &[u8], limits: DecodeLimits) -> Result<Raster, SourceError> {
    let container = parse_container(bytes)?;
    let item = container
        .item(container.primary)
        .ok_or(AvifError::Container(
            "pitm names an item that is not in iinf",
        ))?;
    if &item.kind != b"av01" {
        return Err(AvifError::UnsupportedCodec {
            found: String::from_utf8_lossy(&item.kind).into_owned(),
        }
        .into());
    }

    let (width, height) = container
        .geometry(item)
        .ok_or(AvifError::Container("primary item has no ispe property"))?;
    limits.check_coord(width, height)?;
    limits.check_pixels(width, height)?;

    let config = container
        .av1_config(item)
        .ok_or(AvifError::Container("primary item has no av1C property"))?;
    let has_alpha = container.alpha_item(container.primary).is_some();
    let bands = if has_alpha { 4u64 } else { 3 };
    let sample_bytes = if config.bit_depth() > 8 { 2u64 } else { 1 };
    // Priced from the container's own declared geometry, before a byte of
    // AV1 is decoded. The price is the raster this returns, which is why the
    // band count follows the alpha item and the sample width follows `av1C`.
    limits.check_image_alloc("AVIF frame buffer", width, height, bands, sample_bytes)?;

    let primary = decode_item(&container, item, bytes, limits)?;
    let alpha = match container.alpha_item(container.primary) {
        Some(item) => Some(decode_item(&container, item, bytes, limits)?),
        None => None,
    };
    assemble(&primary, alpha.as_ref(), width, height, limits)
}

/// Gather one item's extents into a contiguous buffer and decode it.
fn decode_item(
    container: &Container,
    item: &Item,
    bytes: &[u8],
    limits: DecodeLimits,
) -> Result<Frame, SourceError> {
    let mut payload = Vec::new();
    for &(offset, length) in &item.extents {
        let end = offset
            .checked_add(length)
            .ok_or(AvifError::Container("iloc extent overflows the file"))?;
        let slice = bytes
            .get(offset..end)
            .ok_or(AvifError::Container("iloc extent points past the file"))?;
        // Every extent is a slice of this file, so a legitimate gather is
        // never longer than the file itself. Anything past that is extent
        // aliasing: an `iloc` with a thousand extents all pointing at the
        // same region amplifies a small file into a large allocation, and
        // that is what this refuses. Deliberately **not** bounded by
        // `max_alloc_bytes`: a small image's compressed payload is routinely
        // larger than its raw frame (`rgb8.avif` is 85 bytes of AV1 for a
        // 36-byte frame), so pricing the payload against the frame budget
        // would refuse legitimate files.
        if payload.len() + slice.len() > bytes.len() {
            return Err(SourceError::AllocLimitExceeded {
                what: "AVIF item payload",
                geometry: None,
                needed_bytes: (payload.len() + slice.len()) as u64,
                max_alloc_bytes: bytes.len() as u64,
            });
        }
        payload.extend_from_slice(slice);
    }
    if payload.is_empty() {
        return Err(AvifError::Container("item has no stored bytes").into());
    }
    let nclx = container.nclx(item);
    decode_av1(&payload, nclx, limits).map_err(Into::into)
}

/// Turn decoded planes into an interleaved [`Raster`].
///
/// The two colour paths are chosen here, and everything not measured is
/// refused rather than approximated.
fn assemble(
    frame: &Frame,
    alpha: Option<&Frame>,
    width: u32,
    height: u32,
    limits: DecodeLimits,
) -> Result<Raster, SourceError> {
    // The container's `ispe` is what the budget was priced against, so a
    // frame that decodes to a different size than it declared is a container
    // that lied and is refused rather than silently re-priced.
    if frame.width != width as usize || frame.height != height as usize {
        return Err(AvifError::Container("decoded frame does not match ispe").into());
    }

    let identity = frame.matrix == MATRIX_IDENTITY;
    let bt601 = matches!(
        frame.matrix,
        MATRIX_BT601_525 | MATRIX_BT601_625 | MATRIX_UNSPECIFIED
    );
    if !frame.full_range || !(identity || bt601 || frame.monochrome) {
        return Err(AvifError::UnsupportedColour {
            matrix: u8::try_from(frame.matrix).unwrap_or(u8::MAX),
            full_range: frame.full_range,
        }
        .into());
    }
    // The measured BT.601 arithmetic is 8-bit. Nothing in the oracle pins a
    // deeper lossy AVIF, so rather than extrapolate the fixed-point constants
    // to 10 or 12 bits this refuses. The identity path has no arithmetic to
    // extrapolate and so carries every depth.
    if bt601 && !identity && !frame.monochrome && frame.bit_depth > 8 {
        return Err(AvifError::UnsupportedColour {
            matrix: u8::try_from(frame.matrix).unwrap_or(u8::MAX),
            full_range: frame.full_range,
        }
        .into());
    }

    // 4:2:2 is refused rather than converted. vips's own `heifsave` cannot
    // write it (`subsample-mode` is on/off, 4:2:0 or 4:4:4), so there is no
    // fixture for it anywhere in the tree, and the two layouts that *are*
    // measured need two different conversions. Guessing which one 4:2:2 uses
    // would be exactly the unchecked claim this module is trying not to make.
    if !frame.monochrome && frame.subsampling_x != frame.subsampling_y {
        return Err(AvifError::UnsupportedColour {
            matrix: u8::try_from(frame.matrix).unwrap_or(u8::MAX),
            full_range: frame.full_range,
        }
        .into());
    }

    let deep = frame.bit_depth > 8;
    let bands = if alpha.is_some() { 4 } else { 3 };
    let format = match (deep, bands) {
        (false, 3) => PixelFormat::Rgb8,
        (false, _) => PixelFormat::Rgba8,
        (true, 3) => PixelFormat::Rgb16,
        (true, _) => PixelFormat::Rgba16,
    };

    let sample_bytes = if deep { 2 } else { 1 };
    let mut out = Vec::with_capacity(frame.width * frame.height * bands * sample_bytes);
    let push = |value: u16, out: &mut Vec<u8>| {
        if deep {
            out.extend_from_slice(&value.to_ne_bytes());
        } else {
            out.push(value as u8);
        }
    };

    for y in 0..frame.height {
        for x in 0..frame.width {
            let luma = frame.luma(x, y);
            let [r, g, b] = if frame.monochrome {
                // `heifload.c:763-765` decodes to RGB unconditionally, so a
                // one-plane AV1 comes back with its luma across all three.
                [luma, luma, luma]
            } else if identity {
                // Identity means the planes already are G, B, R.
                [frame.chroma(2, x, y), luma, frame.chroma(1, x, y)]
            } else {
                // Which op, and not one op, because the two layouts do
                // not agree to the last bit. See `ycbcr_to_rgb_444`.
                let convert = if frame.subsampling_x == 0 && frame.subsampling_y == 0 {
                    ycbcr_to_rgb_444
                } else {
                    ycbcr_to_rgb_420
                };
                let rgb = convert(
                    luma as u8,
                    frame.chroma(1, x, y) as u8,
                    frame.chroma(2, x, y) as u8,
                );
                [u16::from(rgb[0]), u16::from(rgb[1]), u16::from(rgb[2])]
            };
            let justify = |v: u16| {
                if deep {
                    left_justify(v, frame.bit_depth)
                } else {
                    v
                }
            };
            push(justify(r), &mut out);
            push(justify(g), &mut out);
            push(justify(b), &mut out);
            if let Some(alpha) = alpha {
                // The alpha item is a monochrome AV1 of its own, at its own
                // declared depth, so it is justified against *its* bit depth
                // rather than the colour frame's.
                let a = alpha.luma(x.min(alpha.width - 1), y.min(alpha.height - 1));
                push(
                    if deep {
                        left_justify(a, alpha.bit_depth)
                    } else {
                        a
                    },
                    &mut out,
                );
            }
        }
    }

    let mut raster = Raster::new_with_budget(width, height, format, out, limits.max_alloc_bytes)
        .map_err(AvifError::Raster)?;
    raster.meta.interpretation = Some(if deep {
        Interpretation::Rgb16
    } else {
        Interpretation::Srgb
    });
    Ok(raster)
}

/// The `avif`-feature-off body of [`decode_av1`].
///
/// Everything before this point (container parse, codec check, all three
/// decode limits) has already run, so a build without the feature reports a
/// malformed file, an HEVC payload and an over-budget frame exactly as a
/// build with it does. Only the AV1 decode itself is missing, and that is
/// what this says.
#[cfg(not(feature = "avif"))]
fn decode_av1(
    _payload: &[u8],
    _nclx: Option<Nclx>,
    _limits: DecodeLimits,
) -> Result<Frame, AvifError> {
    Err(AvifError::FeatureNotEnabled)
}

/// Decode one AV1 keyframe through `rav1d`.
///
/// # The FFI, and why it is one
///
/// `rav1d` is a pure-Rust port of dav1d, but its **only** public entry points
/// are the `#[no_mangle] extern "C"` dav1d ABI: the safe `rav1d_*` functions
/// beside them are `pub(crate)`, and the context type `Dav1dContext` is an
/// alias for `RawArc<Rav1dContext>` whose inner type is not exported, so it
/// cannot be named from outside the crate at all. Calling it therefore means
/// calling it as C, which is what every other dav1d consumer does.
///
/// The struct layouts below are imported from `rav1d` itself rather than
/// redeclared here, so a field added upstream cannot silently shift an
/// offset. Only the opaque context handle is declared locally, as
/// `*mut c_void`; that is ABI-correct because `RawArc` is
/// `#[repr(transparent)]` over a `NonNull`, so `Option<Dav1dContext>` is a
/// nullable pointer.
#[cfg(feature = "avif")]
fn decode_av1(
    payload: &[u8],
    nclx: Option<Nclx>,
    limits: DecodeLimits,
) -> Result<Frame, AvifError> {
    use rav1d::include::dav1d::data::Dav1dData;
    use rav1d::include::dav1d::dav1d::Dav1dSettings;
    use std::ffi::{c_int, c_void};
    use std::mem::MaybeUninit;

    unsafe extern "C" {
        fn dav1d_default_settings(s: *mut Dav1dSettings);
        fn dav1d_open(c_out: *mut *mut c_void, s: *const Dav1dSettings) -> c_int;
        fn dav1d_data_create(buf: *mut Dav1dData, sz: usize) -> *mut u8;
        fn dav1d_data_unref(buf: *mut Dav1dData);
        fn dav1d_send_data(c: *mut c_void, data: *mut Dav1dData) -> c_int;
        fn dav1d_get_picture(c: *mut c_void, out: *mut Dav1dPicture) -> c_int;
        fn dav1d_picture_unref(p: *mut Dav1dPicture);
        fn dav1d_close(c_out: *mut *mut c_void);
    }

    /// Closes the decoder on every exit path, including an early `?`.
    struct Decoder(*mut c_void);
    impl Drop for Decoder {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: `self.0` came from a successful `dav1d_open` and is
                // closed exactly once, here.
                unsafe { dav1d_close(&mut self.0) };
            }
        }
    }

    /// Releases the picture's reference-counted planes on every exit path.
    struct Picture(Dav1dPicture);
    impl Drop for Picture {
        fn drop(&mut self) {
            // SAFETY: zero-initialised is a valid "empty" picture for
            // `dav1d_picture_unref`, and a filled one came from a successful
            // `dav1d_get_picture`. Either way it is unreffed exactly once.
            unsafe { dav1d_picture_unref(&mut self.0) };
        }
    }

    /// Releases the input buffer on every exit path.
    struct Data(Dav1dData);
    impl Drop for Data {
        fn drop(&mut self) {
            // SAFETY: as `Picture`; zero-initialised is valid to unref.
            unsafe { dav1d_data_unref(&mut self.0) };
        }
    }

    // SAFETY: every call below is the documented dav1d sequence, on pointers
    // this function owns for the whole call. `settings`, `data` and `picture`
    // are stack locals that outlive the calls taking them; the context is
    // owned by `Decoder` and closed in its `Drop`.
    unsafe {
        let mut settings = MaybeUninit::<Dav1dSettings>::zeroed();
        dav1d_default_settings(settings.as_mut_ptr());
        let mut settings = settings.assume_init();
        // The engine above owns the scheduling, for the same reason `exr`
        // drops its `rayon` feature and `jxl-oxide` its threads: a codec that
        // spawns a second pool underneath the tile scheduler is not something
        // this crate wants.
        settings.n_threads = 1;
        settings.max_frame_delay = 1;
        // dav1d's own bomb guard, wired to the caller's ceiling so the
        // decoder refuses an over-large frame before allocating it. This is
        // belt and braces: `decode_avif` already priced `ispe` against the
        // budget, but that is the container's claim and this is the
        // bitstream's.
        settings.frame_size_limit = u32::try_from(limits.max_pixels).unwrap_or(u32::MAX);

        let mut decoder = Decoder(std::ptr::null_mut());
        if dav1d_open(&mut decoder.0, &settings) != 0 || decoder.0.is_null() {
            return Err(AvifError::Decode {
                message: "could not open the AV1 decoder".into(),
            });
        }

        let mut data = Data(MaybeUninit::<Dav1dData>::zeroed().assume_init());
        let buffer = dav1d_data_create(&mut data.0, payload.len());
        if buffer.is_null() {
            return Err(AvifError::Decode {
                message: "could not allocate the AV1 input buffer".into(),
            });
        }
        std::ptr::copy_nonoverlapping(payload.as_ptr(), buffer, payload.len());

        if dav1d_send_data(decoder.0, &mut data.0) != 0 {
            return Err(AvifError::Decode {
                message: "the AV1 decoder refused the payload".into(),
            });
        }

        let mut picture = Picture(MaybeUninit::<Dav1dPicture>::zeroed().assume_init());
        if dav1d_get_picture(decoder.0, &mut picture.0) != 0 {
            return Err(AvifError::Decode {
                message: "the AV1 payload carries no decodable frame".into(),
            });
        }
        read_picture(&picture.0, nclx)
    }
}

/// Copy the decoded planes out of a `Dav1dPicture` into an owned [`Frame`].
///
/// The picture's planes are reference-counted and freed when the caller's
/// `Picture` guard drops, so everything needed later is copied out here
/// rather than borrowed.
///
/// # Safety
///
/// `picture` must be a picture filled by a successful `dav1d_get_picture`,
/// whose plane pointers and strides are therefore valid for its own declared
/// geometry.
#[cfg(feature = "avif")]
unsafe fn read_picture(picture: &Dav1dPicture, nclx: Option<Nclx>) -> Result<Frame, AvifError> {
    let width = usize::try_from(picture.p.w).map_err(|_| AvifError::Decode {
        message: "the AV1 frame declares a negative width".into(),
    })?;
    let height = usize::try_from(picture.p.h).map_err(|_| AvifError::Decode {
        message: "the AV1 frame declares a negative height".into(),
    })?;
    let bit_depth = u8::try_from(picture.p.bpc).unwrap_or(8);
    // 0 is I400 (monochrome), 1 is I420, 2 is I422, 3 is I444.
    let layout = picture.p.layout;
    let (subsampling_x, subsampling_y) = match layout {
        1 => (1u8, 1u8),
        2 => (1, 0),
        _ => (0, 0),
    };
    let monochrome = layout == 0;
    let chroma_width = if monochrome {
        0
    } else {
        width.div_ceil(1 << subsampling_x)
    };
    let chroma_height = if monochrome {
        0
    } else {
        height.div_ceil(1 << subsampling_y)
    };

    // Colour signalling: the container's `colr` box wins where it exists,
    // because that is the file's explicit statement, and the AV1 sequence
    // header is the fallback. `rgb8_icc.avif` is the committed fixture with
    // no `colr` at all, which is what makes the fallback reachable.
    let (matrix, full_range) = match nclx {
        Some(nclx) => (nclx.matrix, nclx.full_range),
        None => match picture.seq_hdr {
            // SAFETY: a filled picture's `seq_hdr` points at the sequence
            // header the decoder parsed, valid for the picture's lifetime.
            Some(header) => {
                let header: &Dav1dSequenceHeader = unsafe { header.as_ref() };
                (
                    u16::try_from(header.mtrx).unwrap_or(u16::MAX),
                    header.color_range != 0,
                )
            }
            None => (MATRIX_UNSPECIFIED, true),
        },
    };

    let plane = |index: usize, w: usize, h: usize| -> Vec<u16> {
        let Some(base) = picture.data[index] else {
            return Vec::new();
        };
        let stride = if index == 0 {
            picture.stride[0]
        } else {
            picture.stride[1]
        };
        let stride = stride as usize;
        let base = base.as_ptr().cast::<u8>();
        let mut out = Vec::with_capacity(w * h);
        for y in 0..h {
            // SAFETY: `base` is the plane's first byte and `stride` its row
            // pitch, both from the picture the decoder filled, so `w` samples
            // from `base + y * stride` are in bounds for `y < h`.
            let row = unsafe { base.add(y * stride) };
            if bit_depth > 8 {
                for x in 0..w {
                    // SAFETY: as above; deep planes hold native-endian u16.
                    out.push(unsafe { row.cast::<u16>().add(x).read_unaligned() });
                }
            } else {
                for x in 0..w {
                    // SAFETY: as above.
                    out.push(u16::from(unsafe { row.add(x).read() }));
                }
            }
        }
        out
    };

    Ok(Frame {
        width,
        height,
        chroma_width,
        chroma_height,
        subsampling_x,
        subsampling_y,
        bit_depth,
        monochrome,
        matrix,
        full_range,
        planes: [
            plane(0, width, height),
            plane(1, chroma_width, chroma_height),
            plane(2, chroma_width, chroma_height),
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lossless 8-bit fixture, which is the bit-exactness anchor for the
    /// whole module: `heifsave --lossless --bitdepth 8` writes AV1 with
    /// `matrix_coefficients = 0` and 4:4:4 chroma, so the planes *are* the
    /// GBR planes and a correct decoder reproduces the source bytes exactly.
    const RGB8: &[u8] = include_bytes!("../oracle-captures/foreign-avif/fixtures/rgb8.avif");
    const RGBA8: &[u8] = include_bytes!("../oracle-captures/foreign-avif/fixtures/rgba8.avif");
    const Q90_444: &[u8] =
        include_bytes!("../oracle-captures/foreign-avif/fixtures/rgb8_q90_444.avif");
    const Q50_420: &[u8] =
        include_bytes!("../oracle-captures/foreign-avif/fixtures/rgb8_q50_420.avif");
    const ODD3X3: &[u8] =
        include_bytes!("../oracle-captures/foreign-avif/fixtures/odd3x3_q50.avif");
    const RGB10: &[u8] = include_bytes!("../oracle-captures/foreign-avif/fixtures/rgb10.avif");
    const RGB12: &[u8] = include_bytes!("../oracle-captures/foreign-avif/fixtures/rgb12.avif");
    const RGB8_ICC: &[u8] =
        include_bytes!("../oracle-captures/foreign-avif/fixtures/rgb8_icc.avif");
    const TRUNCATED: &[u8] =
        include_bytes!("../oracle-captures/foreign-avif/fixtures/truncated.avif");

    /// `vips getpoint` over `rgb8_q50_420.avif`, all twelve pixels, measured
    /// on the pinned 8.18.6 binary with
    /// `env -u VIPS_NOVECTOR vips getpoint rgb8_q50_420.avif X Y`.
    ///
    /// Taken here rather than read out of `oracle.json` because the
    /// `chroma_subsampling` record pins the 4:4:4 fixture's pixels and not
    /// the 4:2:0 one's, and 4:2:0 is the case that needs upsampling.
    const Q50_420_ORACLE: [u8; 36] = [
        0, 26, 13, 25, 103, 90, 180, 160, 87, 105, 85, 12, 66, 144, 131, 146, 224, 211, 111, 91,
        18, 199, 179, 106, 12, 58, 143, 90, 136, 221, 228, 189, 255, 157, 118, 198,
    ];

    /// `oracle.json` -> `eight_bit_lossless_identity.cases.rgb8.source_bytes`.
    const RGB8_ORACLE: [u8; 36] = [
        0, 0, 0, 61, 97, 29, 122, 194, 58, 183, 35, 87, 13, 151, 211, 74, 248, 240, 135, 89, 13,
        196, 186, 42, 26, 46, 166, 87, 143, 195, 148, 240, 224, 209, 81, 253,
    ];
    /// `oracle.json` -> `eight_bit_lossless_identity.cases.rgba8.source_bytes`.
    const RGBA8_ORACLE: [u8; 48] = [
        0, 0, 0, 0, 61, 97, 29, 85, 122, 194, 58, 170, 183, 35, 87, 255, 13, 151, 211, 40, 74, 248,
        240, 125, 135, 89, 13, 210, 196, 186, 42, 39, 26, 46, 166, 80, 87, 143, 195, 165, 148, 240,
        224, 250, 209, 81, 253, 79,
    ];
    /// `oracle.json` -> `chroma_subsampling.fixture_444_getpoint`, flattened.
    const Q90_444_ORACLE: [u8; 36] = [
        9, 0, 0, 62, 97, 33, 122, 191, 66, 184, 33, 86, 16, 148, 210, 69, 253, 229, 139, 88, 25,
        193, 189, 40, 30, 46, 167, 83, 145, 186, 144, 244, 220, 205, 84, 251,
    ];
    /// `oracle.json` -> `odd_dimensions.getpoint`, flattened.
    const ODD3X3_ORACLE: [u8; 27] = [
        0, 16, 36, 42, 100, 120, 150, 178, 58, 75, 133, 153, 142, 200, 220, 84, 112, 0, 13, 59,
        137, 92, 138, 216, 144, 243, 223,
    ];
    /// `oracle.json` -> `bit_depth_carrier.by_bitdepth."10".read_back`.
    const RGB10_ORACLE: [u16; 36] = [
        0, 0, 192, 256, 960, 1024, 4032, 4096, 32704, 32768, 65472, 65472, 0, 0, 192, 256, 960,
        1024, 4032, 4096, 32704, 32768, 65472, 65472, 0, 0, 192, 256, 960, 1024, 4032, 4096, 32704,
        32768, 65472, 65472,
    ];
    /// `oracle.json` -> `bit_depth_carrier.by_bitdepth."12".read_back`.
    const RGB12_ORACLE: [u16; 36] = [
        0, 0, 240, 256, 1008, 1024, 4080, 4096, 32752, 32768, 65472, 65520, 0, 0, 240, 256, 1008,
        1024, 4080, 4096, 32752, 32768, 65472, 65520, 0, 0, 240, 256, 1008, 1024, 4080, 4096,
        32752, 32768, 65472, 65520,
    ];

    /// The decoded bytes of `bytes`, or the failure, at default limits.
    fn decode(bytes: &[u8]) -> Result<Raster, SourceError> {
        decode_avif(bytes, DecodeLimits::default())
    }

    /// Every 8-bit sample of a decode, interleaved, as the raster holds them.
    fn samples8(raster: &Raster) -> Vec<u8> {
        raster.data().to_vec()
    }

    /// Every 16-bit sample of a decode, in the host order the raster stores.
    fn samples16(raster: &Raster) -> Vec<u16> {
        raster
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_ne_bytes(*c))
            .collect()
    }

    /*
     * THE test. AV1 lossless with `matrix_coefficients = 0` and 4:4:4 stores
     * the RGB planes as-is, so there is no colour conversion to disagree
     * about and a correct decoder reproduces the encoder's input byte for
     * byte. This is the claim that makes the whole module worth having: not
     * "close to vips" but equal to it, with no tolerance anywhere.
     * Input: `rgb8.avif` -> Output: exactly the 36 bytes the oracle recorded
     * as this fixture's source.
     */
    #[cfg_attr(not(feature = "avif"), ignore = "needs the avif feature")]
    #[test]
    fn lossless_identity_is_bit_exact_against_the_oracle() {
        let raster = decode(RGB8).expect("rgb8 decodes");
        assert_eq!((raster.width(), raster.height()), (4, 3));
        assert_eq!(raster.format(), PixelFormat::Rgb8);
        assert_eq!(
            samples8(&raster),
            RGB8_ORACLE,
            "a lossless identity AVIF must round-trip exactly, not approximately"
        );
    }

    /*
     * The alpha plane is a second, monochrome AV1 item, so a decoder that
     * ignored `auxl` would return a perfectly plausible three-band image and
     * silently drop transparency. Pinned against the oracle's own four-band
     * source bytes.
     * Input: `rgba8.avif` -> Output: 4 bands, exactly the 48 recorded bytes.
     */
    #[cfg_attr(not(feature = "avif"), ignore = "needs the avif feature")]
    #[test]
    fn the_alpha_item_becomes_the_fourth_band_exactly() {
        let raster = decode(RGBA8).expect("rgba8 decodes");
        assert_eq!(raster.format(), PixelFormat::Rgba8, "alpha makes it RGBA");
        assert_eq!(samples8(&raster), RGBA8_ORACLE);
    }

    /*
     * The lossy 4:4:4 path, which is where the YCbCr conversion runs with no
     * chroma upsampling in the way. This isolates the matrix arithmetic: if
     * this passes and the 4:2:0 test below fails, the upsampler is wrong, and
     * if both fail the matrix is.
     * Input: `rgb8_q90_444.avif` -> Output: the twelve pixels vips returns.
     */
    #[cfg_attr(not(feature = "avif"), ignore = "needs the avif feature")]
    #[test]
    fn lossy_444_matches_the_oracle_pixel_for_pixel() {
        let raster = decode(Q90_444).expect("q90 444 decodes");
        assert_eq!(samples8(&raster), Q90_444_ORACLE);
    }

    /*
     * The 4:2:0 path, which needs nearest-neighbour chroma upsampling on top
     * of the same matrix. The oracle's own note calls this "the axis a
     * decoder is most likely to get subtly wrong", and it is: a bilinear
     * upsampler passes nothing here, and a float matrix gets eleven of these
     * twelve pixels and misses the twelfth by one.
     * Input: `rgb8_q50_420.avif` -> Output: the twelve pixels vips returns.
     */
    #[cfg_attr(not(feature = "avif"), ignore = "needs the avif feature")]
    #[test]
    fn nearest_neighbour_chroma_matches_the_oracle() {
        let raster = decode(Q50_420).expect("q50 420 decodes");
        assert_eq!(samples8(&raster), Q50_420_ORACLE);
    }

    /*
     * A 3x3 4:2:0 image has 2x2 chroma planes, so the right column and the
     * bottom row are half-covered and `x >> 1` reaches the plane width. The
     * oracle pinned this fixture precisely because it is "where a hand-rolled
     * upsampler diverges first and the header gives no hint that it has".
     * Input: `odd3x3_q50.avif` -> Output: the nine pixels vips returns.
     */
    #[cfg_attr(not(feature = "avif"), ignore = "needs the avif feature")]
    #[test]
    fn odd_dimensions_do_not_run_off_the_chroma_plane() {
        let raster = decode(ODD3X3).expect("odd 3x3 decodes");
        assert_eq!((raster.width(), raster.height()), (3, 3));
        assert_eq!(samples8(&raster), ODD3X3_ORACLE);
    }

    /*
     * Deeper bit depths left-justify: `heifload.c:1000-1016` shifts the
     * sample up by `16 - bits_per_pixel`, so the low bits are always zero and
     * the maximum is 65472 at 10 bits and 65520 at 12. A decoder that
     * returned the raw 0..1023 samples, or that rescaled to full 16-bit
     * range, would be wrong in a way that looks almost right on a histogram.
     * Input: `rgb10.avif` and `rgb12.avif` -> Output: the oracle's own
     * `read_back` arrays, and the two ceilings it recorded.
     */
    #[cfg_attr(not(feature = "avif"), ignore = "needs the avif feature")]
    #[test]
    fn deep_samples_left_justify() {
        for (bytes, expected, low_zero_bits, ceiling) in [
            (RGB10, &RGB10_ORACLE[..], 6u32, 65472u16),
            (RGB12, &RGB12_ORACLE[..], 4, 65520),
        ] {
            let raster = decode(bytes).expect("deep fixture decodes");
            assert_eq!(raster.format(), PixelFormat::Rgb16);
            let got = samples16(&raster);
            assert_eq!(got, expected, "left-justified samples must match vips");
            assert!(
                got.iter()
                    .all(|s| s.trailing_zeros() >= low_zero_bits || *s == 0),
                "left justification leaves the low {low_zero_bits} bits zero"
            );
            assert!(
                got.iter().max().is_some_and(|&m| m <= ceiling),
                "the maximum representable sample is {ceiling}"
            );
        }
    }

    /*
     * `rgb8_icc.avif` carries no `colr` box at all, so the matrix has to come
     * from the AV1 sequence header instead. Without that fallback the decode
     * either refuses a perfectly good file or converts it with the wrong
     * assumption; this is the only committed fixture that reaches the
     * fallback, which is why it is worth its own test rather than a decode
     * smoke check.
     * Input: `rgb8_icc.avif` -> Output: it decodes, at the right geometry.
     */
    #[cfg_attr(not(feature = "avif"), ignore = "needs the avif feature")]
    #[test]
    fn a_file_with_no_colr_box_reads_its_matrix_from_the_sequence_header() {
        let container = parse_container(RGB8_ICC).expect("parses");
        let item = container.item(container.primary).expect("primary");
        assert!(
            container.nclx(item).is_none(),
            "this fixture is only interesting because it has no nclx colr"
        );
        let raster = decode(RGB8_ICC).expect("it still decodes");
        assert_eq!((raster.width(), raster.height()), (4, 3));
    }

    /*
     * AVIF, HEIF and boxed JPEG XL are all ISOBMFF, so their magics live in
     * the same twelve bytes and the sniff table has to tell them apart
     * without either shadowing the other. Both directions, because a
     * one-directional test passes just as happily when a row is unreachable
     * as when it is correct.
     * Input: an AVIF, a boxed JXL, a bare JXL codestream -> Output: each
     * sniffs as itself and none as another.
     */
    #[test]
    fn avif_and_boxed_jpeg_xl_do_not_shadow_each_other() {
        use crate::source::{SniffedFormat, sniff};

        assert_eq!(
            sniff(&RGB8[..16]),
            Some(SniffedFormat::Avif),
            "a real AVIF must sniff as AVIF"
        );

        // The boxed JPEG XL signature box: a 12-byte box whose type is
        // `JXL ` and whose payload is the line-ending check.
        let boxed_jxl = b"\x00\x00\x00\x0cJXL \x0d\x0a\x87\x0a\x00\x00\x00\x00";
        assert_eq!(
            sniff(boxed_jxl),
            Some(SniffedFormat::Jxl),
            "a boxed JPEG XL must still sniff as JPEG XL, not as AVIF"
        );
        let bare_jxl = b"\xff\x0a\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00";
        assert_eq!(sniff(bare_jxl), Some(SniffedFormat::Jxl));

        // The sharp case: an ISOBMFF file whose `ftyp` box is exactly 12
        // bytes long opens with the same four bytes as the JPEG XL box
        // header. Only the type at offset 4 separates them.
        let mut twelve_byte_ftyp = *b"\x00\x00\x00\x0cftypavif\x00\x00\x00\x00";
        assert_eq!(
            sniff(&twelve_byte_ftyp),
            Some(SniffedFormat::Avif),
            "a 12-byte ftyp box shares JPEG XL's first four bytes and is still AVIF"
        );
        // ... and mutating only the brand takes it out of the table entirely,
        // which is what proves the match is on the brand and not on `ftyp`.
        twelve_byte_ftyp[8..12].copy_from_slice(b"heic");
        assert_eq!(
            sniff(&twelve_byte_ftyp),
            None,
            "HEIC is not claimed: its payload is usually HEVC, which this cannot decode"
        );
    }

    /*
     * `avis` is the image-sequence brand and is out of scope, so it must not
     * be picked up by the sniffer even though the committed fixture behind it
     * carries a perfectly decodable AV1 still. The oracle recorded vips
     * routing this one to `magickload` rather than `heifload` for its own
     * reasons; here it simply is not claimed.
     * Input: `brand_avis.avif` -> Output: not sniffed as AVIF.
     */
    #[test]
    fn the_image_sequence_brand_is_not_claimed() {
        use crate::source::sniff;
        const AVIS: &[u8] =
            include_bytes!("../oracle-captures/foreign-avif/fixtures/brand_avis.avif");
        assert_eq!(&AVIS[4..12], b"ftypavis", "the fixture is what this claims");
        assert_eq!(
            sniff(&AVIS[..16]),
            None,
            "the image-sequence brand is deliberately out of scope"
        );
    }

    /*
     * A truncated file is attacker-shaped input: every length in an ISOBMFF
     * header is read from the file itself, so a short read has to be a typed
     * refusal and never a panic or an out-of-bounds slice.
     * Input: `truncated.avif` -> Output: a typed container error.
     */
    #[test]
    fn a_truncated_file_is_a_typed_refusal_not_a_panic() {
        let err = decode(TRUNCATED).expect_err("a truncated file must be refused");
        assert!(
            matches!(err, SourceError::Avif(AvifError::Container(_))),
            "expected a container error, got {err:?}"
        );
    }

    /*
     * The #498 wall, reported by name. An HEIC file parses as a perfectly
     * good container right up to the codec check, so without this the failure
     * would surface as some downstream confusion rather than as "this needs
     * an HEVC decoder, which is the thing that does not exist".
     * Input: `rgb8.avif` with its primary item retyped `hvc1` -> Output:
     * `UnsupportedCodec` naming what it found.
     */
    #[test]
    fn an_hevc_payload_is_refused_by_name() {
        let mut bytes = RGB8.to_vec();
        let at = bytes
            .windows(4)
            .position(|w| w == b"av01")
            .expect("the fixture declares an av01 item");
        bytes[at..at + 4].copy_from_slice(b"hvc1");
        let err = decode(&bytes).expect_err("HEVC must be refused");
        match err {
            SourceError::Avif(AvifError::UnsupportedCodec { found }) => {
                assert_eq!(found, "hvc1");
            }
            other => panic!("expected UnsupportedCodec, got {other:?}"),
        }
    }

    /*
     * The three decode ceilings, all of them applied to the container's
     * declared `ispe` before a byte of AV1 is decoded. That ordering is the
     * point: AVIF is a compressed container, so a 355-byte file can declare a
     * frame far larger than any budget, and a ceiling checked after the
     * decode is a ceiling that has already lost.
     * Input: `rgb8.avif` at each ceiling set below what it needs -> Output:
     * the matching typed refusal, and a decode at the ceiling itself.
     */
    #[cfg_attr(not(feature = "avif"), ignore = "needs the avif feature")]
    #[test]
    fn every_decode_ceiling_is_applied_to_the_declared_geometry() {
        // 4x3x3 bands x 1 byte.
        let price = 36;
        let exact = DecodeLimits::default().with_max_alloc_bytes(price);
        assert!(
            decode_avif(RGB8, exact).is_ok(),
            "the budget is enough at exactly the price"
        );
        let tight = DecodeLimits::default().with_max_alloc_bytes(price - 1);
        match decode_avif(RGB8, tight) {
            Err(SourceError::AllocLimitExceeded {
                what, needed_bytes, ..
            }) => {
                assert_eq!(what, "AVIF frame buffer");
                assert_eq!(needed_bytes, price);
            }
            other => panic!("expected AllocLimitExceeded, got {other:?}"),
        }

        let narrow = DecodeLimits::default().with_max_coord(3);
        assert!(
            matches!(
                decode_avif(RGB8, narrow),
                Err(SourceError::CoordLimitExceeded { .. })
            ),
            "a 4-wide frame must trip a 3-pixel coordinate ceiling"
        );

        let few = DecodeLimits::default().with_max_pixels(11);
        assert!(
            matches!(
                decode_avif(RGB8, few),
                Err(SourceError::DimensionLimitExceeded { .. })
            ),
            "12 pixels must trip an 11-pixel ceiling"
        );
    }

    /*
     * The ceilings answer the same in a build without the feature, because
     * everything up to the AV1 decode runs either way. Without this a
     * default build could quietly stop pricing AVIF at all and no test in
     * the tree would notice.
     * Input: `rgb8.avif` one byte under its price -> Output: the same
     * `AllocLimitExceeded`, feature or no feature.
     */
    #[test]
    fn the_budget_refusal_does_not_depend_on_the_avif_feature() {
        let tight = DecodeLimits::default().with_max_alloc_bytes(35);
        let err = decode_avif(RGB8, tight).expect_err("must be refused either way");
        assert!(
            err.is_alloc_limit(),
            "the budget refusal must be recognisable in both builds: {err:?}"
        );
    }

    /*
     * Only two colour encodings are measured, so a third has to be refused
     * rather than converted by a formula nothing has checked. Limited-range
     * video is the reachable case: flipping the `colr` box's full-range bit
     * is a one-byte edit to a real file.
     * Input: `rgb8.avif` with full_range cleared -> Output: refused, naming
     * the matrix and the range.
     */
    #[cfg_attr(not(feature = "avif"), ignore = "needs the avif feature")]
    #[test]
    fn an_unmeasured_colour_encoding_is_refused_rather_than_guessed() {
        let mut bytes = RGB8.to_vec();
        let at = bytes
            .windows(4)
            .position(|w| w == b"nclx")
            .expect("the fixture carries an nclx colr box");
        // The full-range flag is the top bit of the byte after the three
        // 16-bit CICP codes.
        bytes[at + 10] = 0;
        match decode(&bytes) {
            Err(SourceError::Avif(AvifError::UnsupportedColour { full_range, .. })) => {
                assert!(!full_range, "the refusal reports the range it saw");
            }
            other => panic!("expected UnsupportedColour, got {other:?}"),
        }
    }

    /*
     * The container walk is what this module had to hand-roll, because the
     * obvious crate for it (`avif-parse`) is MPL-2.0 and taking it would have
     * reversed the licence decision #502 was made on. So it gets tested
     * directly rather than only through a successful decode: a decode that
     * works proves the happy path, and says nothing about which of `iloc`,
     * `ipma` and `iinf` were actually read correctly.
     * Input: `rgb8.avif` -> Output: one `av01` item, primary, 4x3, with the
     * `av1C` and `colr` the oracle recorded for it.
     */
    #[test]
    fn the_container_walk_finds_the_primary_item_and_its_properties() {
        let container = parse_container(RGB8).expect("rgb8 parses");
        assert_eq!(container.primary, 1, "pitm names item 1");
        let item = container.item(1).expect("item 1 is in iinf");
        assert_eq!(&item.kind, b"av01", "the primary item is an AV1 image");
        // The oracle's own `iloc` line for this file reads `items=1@270+85`.
        assert_eq!(
            item.extents,
            vec![(270, 85)],
            "iloc places the payload where the oracle recorded it"
        );
        assert_eq!(
            container.geometry(item),
            Some((4, 3)),
            "ispe declares the geometry the oracle recorded"
        );
        assert_eq!(
            container.nclx(item),
            Some(Nclx {
                matrix: 0,
                full_range: true
            }),
            "colr declares the identity matrix at full range"
        );
        let config = container.av1_config(item).expect("av1C is associated");
        assert_eq!(config.bit_depth(), 8);
        assert!(
            !config.subsampling_x && !config.subsampling_y,
            "lossless is 4:4:4, because heifsave overrides subsampling off"
        );
    }

    /*
     * `ipma` indexes properties by 1-based position in `ipco`, so a parser
     * that skipped the boxes it does not understand would shift every index
     * after them and hand the wrong property to the item. `rgb8.avif` carries
     * a `pixi` box this module does not read, sitting between `ispe` and the
     * end, which is exactly the shape that breaks.
     * Input: `rgb8.avif`'s `ipco` -> Output: four properties, with the
     * unread one held as a placeholder rather than dropped.
     */
    #[test]
    fn unread_properties_keep_their_slot_so_ipma_indices_stay_aligned() {
        let container = parse_container(RGB8).expect("rgb8 parses");
        assert_eq!(
            container.properties.len(),
            4,
            "av1C, colr, ispe and pixi are all four in ipco"
        );
        assert!(
            container.properties.contains(&Property::Other),
            "pixi is not read, but it still occupies its slot"
        );
    }

    /*
     * The alpha plane is a second AV1 item, monochrome, joined to the image
     * by an `auxl` reference and marked by an `auxC` URN. Both conditions are
     * required: an `auxl` alone could point at a depth map or a gain map, and
     * treating one as alpha would put the wrong plane in the fourth band.
     * Input: `rgba8.avif` -> Output: item 2 found as the alpha of item 1.
     */
    #[test]
    fn the_alpha_plane_is_found_through_auxl_and_its_urn() {
        let container = parse_container(RGBA8).expect("rgba8 parses");
        let alpha = container
            .alpha_item(1)
            .expect("rgba8 carries an alpha item");
        assert_eq!(alpha.id, 2, "the oracle records the alpha as item 2");
        let config = container.av1_config(alpha).expect("the alpha has an av1C");
        assert!(config.monochrome, "an alpha plane is a monochrome AV1");
        assert!(
            container.alpha_item(2).is_none(),
            "the alpha item does not itself have an alpha"
        );
    }
}
