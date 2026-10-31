//! Red stub (issue #1330): the legal tile table is not written yet.

use thiserror::Error;

use super::pyramid::{RasterChannels, RasterCodec};
use crate::pixel::PixelFormat;
use crate::sink::TileFormat;

/// Why a tile is not one PVec stores.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RasterTileError {
    /// Stub.
    #[error("raw")]
    RawTiles,
    /// Stub.
    #[error("jpeg alpha")]
    JpegAlpha,
    /// Stub.
    #[error("16 bit {0:?}")]
    SixteenBit(PixelFormat),
    /// Stub.
    #[error("format {0:?}")]
    PixelFormat(PixelFormat),
    /// Stub.
    #[error("bit depth {depth} for {format:?}")]
    BitDepth {
        /// Stub.
        format: PixelFormat,
        /// Stub.
        depth: u8,
    },
    /// Stub.
    #[error("edge {0}")]
    TileEdge(u32),
    /// Stub.
    #[error("too long")]
    TooLong {
        /// Stub.
        len: u64,
        /// Stub.
        cap: u64,
        /// Stub.
        edge: u32,
    },
}

/// Stub.
#[must_use]
pub fn max_tile_bytes(_edge: u32) -> u64 {
    0
}

/// Stub.
///
/// # Errors
///
/// Always, for now.
pub fn check_tile_len(_edge: u32, _len: usize) -> Result<(), RasterTileError> {
    Err(RasterTileError::RawTiles)
}

/// Stub.
///
/// # Errors
///
/// Always, for now.
pub fn check_tile_edge(edge: u32) -> Result<(), RasterTileError> {
    Err(RasterTileError::TileEdge(edge))
}

/// Stub.
///
/// # Errors
///
/// Always, for now.
pub fn tile_tags(
    _format: TileFormat,
    _pixel: PixelFormat,
    _bit_depth: u8,
) -> Result<(RasterCodec, RasterChannels), RasterTileError> {
    Err(RasterTileError::RawTiles)
}

/// Stub.
#[must_use]
pub fn render_markdown() -> String {
    String::new()
}
