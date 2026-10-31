//! The tiles PVec stores: the legal table and LIMIT-15 (issue #1330, RL1.1).
//!
//! PVec never decodes a pixel. It reads a tile's header with a bounded sniffer and refuses a
//! tile that does not say `tile_edge` by `tile_edge` in the placement's channels and bit depth
//! (RASTER-03, RASTER-04, `docs/RASTER.md` section 9 in libviprs/pvec). This module is the
//! producer's side of that contract:
//!
//! * [`tile_tags`] maps `(tile format, pixel format, bit depth)` to the codec and channels PVec
//!   stores, or names the reason there is no such row. [`render_markdown`] writes the same table
//!   to `docs/pvec-raster-tiles.md`, and a test fails if the file drifts.
//! * [`check_tile_len`] is LIMIT-15 at encode time: a stored tile longer than
//!   `2 * 4 * edge^2 + 65,536` bytes is refused here, before it reaches PVec's writer.
//! * [`check_tile_edge`] is RASTER-01's tile edge.
//!
//! Two PNG forms libviprs can write are deliberately **not** rows: an Adam7-interlaced PNG
//! (`Raster::encode_png_interlaced`) and a palette PNG (`Raster::encode_png_palette`). The
//! pyramid never calls either, and a test reads every tile it makes to prove it. A 1-bit tile is
//! an encoding of `Gray8` ([`Raster::encode_png_bilevel`](crate::Raster::encode_png_bilevel)),
//! so it is the depth column of a row and not a new `PixelFormat`.

use std::fmt::Write as _;
use std::mem::discriminant;

use thiserror::Error;

use super::pyramid::{RasterChannels, RasterCodec};
use crate::pixel::PixelFormat;
use crate::sink::TileFormat;

/// Why a tile is not one PVec stores.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RasterTileError {
    /// Raw samples carry neither dimensions nor a pixel format (RASTER-04).
    #[error("raw tiles carry no dimensions or pixel format, so PVec cannot store them")]
    RawTiles,
    /// JPEG has nowhere to put alpha (RASTER-04); flatten first, or use PNG or WebP.
    #[error("JPEG cannot carry alpha; flatten the tile or choose PNG or WebP")]
    JpegAlpha,
    /// PVec stores 1 and 8 bit samples; the producer converts first and reports it.
    #[error("{0:?} is 16-bit; convert to 8 bits before tiling")]
    SixteenBit(PixelFormat),
    /// The pixel format has no tile encoding (float, multiband).
    #[error("{0:?} has no tile encoding; convert it to gray, RGB or RGBA at 8 bits")]
    PixelFormat(PixelFormat),
    /// A bit depth the pair has no row for (1 bit is gray PNG only).
    #[error("bit depth {depth} is not available for {format:?} in this codec")]
    BitDepth {
        /// The pixel format asked for.
        format: PixelFormat,
        /// The depth asked for.
        depth: u8,
    },
    /// A tile edge other than 256, 512 or 1024 (RASTER-01).
    #[error("tile edge {0} is not 256, 512 or 1024")]
    TileEdge(u32),
    /// The stored bytes exceed LIMIT-15.
    #[error("a {edge} px tile of {len} bytes exceeds LIMIT-15 ({cap} bytes)")]
    TooLong {
        /// The tile's stored length.
        len: u64,
        /// `2 * 4 * edge^2 + 65,536`.
        cap: u64,
        /// The placement's tile edge.
        edge: u32,
    },
}

/// The tile edges PVec accepts (RASTER-01).
const EDGES: [u32; 3] = [256, 512, 1024];

/// A row of the legal table: what the producer is asked for and what PVec stores.
struct Row {
    format: TileFormat,
    pixel: PixelFormat,
    depth: u8,
    codec: RasterCodec,
    channels: RasterChannels,
}

const fn row(
    format: TileFormat,
    pixel: PixelFormat,
    depth: u8,
    codec: RasterCodec,
    channels: RasterChannels,
) -> Row {
    Row {
        format,
        pixel,
        depth,
        codec,
        channels,
    }
}

const JPEG: TileFormat = TileFormat::Jpeg { quality: 0 };

/// Every `(tile format, pixel format, bit depth)` PVec stores, as the producer writes it.
const TABLE: [Row; 9] = [
    row(
        TileFormat::Png,
        PixelFormat::Gray8,
        1,
        RasterCodec::Png,
        RasterChannels::Gray,
    ),
    row(
        TileFormat::Png,
        PixelFormat::Gray8,
        8,
        RasterCodec::Png,
        RasterChannels::Gray,
    ),
    row(
        TileFormat::Png,
        PixelFormat::Rgb8,
        8,
        RasterCodec::Png,
        RasterChannels::Rgb,
    ),
    row(
        TileFormat::Png,
        PixelFormat::Rgba8,
        8,
        RasterCodec::Png,
        RasterChannels::Rgba,
    ),
    row(
        JPEG,
        PixelFormat::Gray8,
        8,
        RasterCodec::Jpeg,
        RasterChannels::Gray,
    ),
    row(
        JPEG,
        PixelFormat::Rgb8,
        8,
        RasterCodec::Jpeg,
        RasterChannels::Rgb,
    ),
    row(
        TileFormat::Webp,
        PixelFormat::Gray8,
        8,
        RasterCodec::Webp,
        RasterChannels::Gray,
    ),
    row(
        TileFormat::Webp,
        PixelFormat::Rgb8,
        8,
        RasterCodec::Webp,
        RasterChannels::Rgb,
    ),
    row(
        TileFormat::Webp,
        PixelFormat::Rgba8,
        8,
        RasterCodec::Webp,
        RasterChannels::Rgba,
    ),
];

/// The codec and channels PVec stores for a tile written in `format` from `pixel` samples at
/// `bit_depth` (1 or 8).
///
/// # Errors
///
/// A [`RasterTileError`] names why there is no row: raw tiles, JPEG with alpha, 16-bit samples,
/// a pixel format with no tile encoding, or a depth the pair does not have.
pub fn tile_tags(
    format: TileFormat,
    pixel: PixelFormat,
    bit_depth: u8,
) -> Result<(RasterCodec, RasterChannels), RasterTileError> {
    if matches!(format, TileFormat::Raw) {
        return Err(RasterTileError::RawTiles);
    }
    if let Some(r) = TABLE.iter().find(|r| {
        discriminant(&r.format) == discriminant(&format) && r.pixel == pixel && r.depth == bit_depth
    }) {
        return Ok((r.codec, r.channels));
    }
    match pixel {
        PixelFormat::Gray16 | PixelFormat::Rgb16 | PixelFormat::Rgba16 => {
            Err(RasterTileError::SixteenBit(pixel))
        }
        PixelFormat::Rgba8 if matches!(format, TileFormat::Jpeg { .. }) => {
            Err(RasterTileError::JpegAlpha)
        }
        PixelFormat::Gray8 | PixelFormat::Rgb8 | PixelFormat::Rgba8 => {
            Err(RasterTileError::BitDepth {
                format: pixel,
                depth: bit_depth,
            })
        }
        other => Err(RasterTileError::PixelFormat(other)),
    }
}

/// RASTER-01's tile edge.
///
/// # Errors
///
/// [`RasterTileError::TileEdge`] for anything but 256, 512 or 1024.
pub fn check_tile_edge(edge: u32) -> Result<(), RasterTileError> {
    if EDGES.contains(&edge) {
        Ok(())
    } else {
        Err(RasterTileError::TileEdge(edge))
    }
}

/// LIMIT-15: the most bytes PVec stores for one tile, `2 * 4 * edge^2 + 65,536` (589,824 at 256,
/// 2,162,688 at 512, 8,454,144 at 1024). Twice the raw RGBA size plus a header allowance, which
/// an honest PNG, JPEG or WebP never reaches.
#[must_use]
pub const fn max_tile_bytes(edge: u32) -> u64 {
    2 * 4 * (edge as u64) * (edge as u64) + 65_536
}

/// LIMIT-15 at encode time.
///
/// # Errors
///
/// [`RasterTileError::TooLong`] when `len` is above [`max_tile_bytes`].
pub fn check_tile_len(edge: u32, len: usize) -> Result<(), RasterTileError> {
    let cap = max_tile_bytes(edge);
    if len as u64 > cap {
        return Err(RasterTileError::TooLong {
            len: len as u64,
            cap,
            edge,
        });
    }
    Ok(())
}

/// `docs/pvec-raster-tiles.md`, rendered from the table.
#[must_use]
pub fn render_markdown() -> String {
    let codec_id = |c: RasterCodec| match c {
        RasterCodec::Png => (1, "PNG"),
        RasterCodec::Jpeg => (2, "JPEG"),
        RasterCodec::Webp => (3, "WebP"),
    };
    let channel_id = |c: RasterChannels| match c {
        RasterChannels::Gray => (0, "gray"),
        RasterChannels::Rgb => (2, "rgb"),
        RasterChannels::Rgba => (3, "rgba"),
    };
    let mut out = String::new();
    out.push_str(
        "# Tiles PVec stores (RL1.1, issue #1330)\n\n\
         This file is generated from `TABLE` in `src/raster_pvec/tiles.rs`. Do not edit it by hand: \
         a test renders the table and fails if the committed file differs by a byte. To change it, \
         change the table and run \
         `PVEC_RASTER_TILES_BLESS=1 cargo test --test pvec_1330_raster_tiles`.\n\n\
         A row is a tile libviprs writes for a PVec raster, and the tags PVec stores for it \
         (`tile_codec` and `pixel_format` of `docs/RASTER.md` section 1). The bit depth is 1 or 8. \
         A 1-bit tile is an encoding of `Gray8` (`Raster::encode_png_bilevel`), not a pixel format.\n\n\
         | tile format | source pixels | bit depth | `tile_codec` | `pixel_format` |\n\
         |---|---|---|---|---|\n",
    );
    for r in &TABLE {
        let format = match r.format {
            TileFormat::Png => "PNG",
            TileFormat::Webp => "WebP",
            _ => "JPEG",
        };
        let (cid, cname) = codec_id(r.codec);
        let (pid, pname) = channel_id(r.channels);
        let _ = writeln!(
            out,
            "| {format} | `{:?}` | {} | {cid} ({cname}) | {pid} ({pname}) |",
            r.pixel, r.depth
        );
    }
    out.push_str(
        "\nEverything else is a typed `RasterTileError`:\n\n\
         | refusal | when |\n|---|---|\n\
         | `RawTiles` | `TileFormat::Raw`: no dimensions, no pixel format (RASTER-04) |\n\
         | `JpegAlpha` | JPEG from `Rgba8`; flatten first or use PNG or WebP (RASTER-04) |\n\
         | `SixteenBit` | `Gray16`, `Rgb16`, `Rgba16`: the producer converts to 8 bits first and reports `converted-bit-depth` |\n\
         | `PixelFormat` | float and multiband sources |\n\
         | `BitDepth` | a depth the pair has no row for; 1 bit is gray PNG only |\n\
         | `TileEdge` | an edge other than 256, 512 or 1024 (RASTER-01) |\n\
         | `TooLong` | a stored tile above `2 * 4 * edge^2 + 65,536` bytes: 589,824 at 256, 2,162,688 at 512, 8,454,144 at 1024 (LIMIT-15) |\n\n\
         Interlaced PNG and palette PNG are never produced on this path, though `Raster::encode_png_interlaced` \
         and `Raster::encode_png_palette` exist: PVec's sniffer refuses both.\n",
    );
    out
}
