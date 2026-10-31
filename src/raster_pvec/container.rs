//! The hand-off of a raster pyramid to PVec's writer (issue #1328, R2.4).
//!
//! [`write_rasters`] gives PVec the placements and the **encoded tile bytes, untouched**, each
//! with its `(level, col, row)`; PVec stores them as they are and never decodes a pixel (ADR 0018,
//! decision 3). A tile the producer did not store (`skip_blanks`, or padding past a level's grid)
//! is simply not handed over, which PVec reads back as a not-stored tile.
//!
//! The writer refuses what the reader would (RASTER-01 to RASTER-07), so a producer learns at
//! write time: that refusal comes back as [`RasterWriteError::Pvec`] with PVec's own error as the
//! `source()`.
//!
//! Needs the `pvec-raster` feature. This is the only module that names the PVec facade crate.

use pvec_facade::PvecWriter;
use pvec_facade::unstable::{Band, PixelFormat, Placement, RasterId, TileCodec, TileIn};
use thiserror::Error;

use super::pyramid::{RasterBand, RasterChannels, RasterCodec, RasterPyramid};

/// Why a pyramid could not be written into a PVec.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RasterWriteError {
    /// More rasters than PVec allows in one artifact (LIMIT-13).
    #[error("PVec takes at most 4096 rasters in one artifact, got {0}")]
    TooMany(usize),
    /// The pyramid does not fit the placement's integer fields.
    #[error("the pyramid does not fit a PVec placement: {0}")]
    Shape(&'static str),
    /// PVec's writer refused a placement or a tile.
    #[error("PVec's writer refused the raster")]
    Pvec(#[source] pvec_facade::PvecError),
}

/// Write `pyramids` into `writer` as the artifact's raster stream, in order: the first gets
/// `RasterId` 0, the next 1, and so on. The band is each pyramid's own (the underlay unless a PDF
/// image was painted after every path) and the opacity 255.
///
/// Returns the ids, which are the positions in `pyramids`.
///
/// # Errors
///
/// [`RasterWriteError`]: too many rasters, a pyramid whose size does not fit the placement, or a
/// refusal by PVec's writer (a singular placement map, an attribution that breaks RASTER-07, a
/// tile the sniffer rejects) with PVec's error as the source.
pub fn write_rasters<'a>(
    writer: &mut PvecWriter,
    pyramids: impl IntoIterator<Item = &'a RasterPyramid>,
) -> Result<Vec<RasterId>, RasterWriteError> {
    let pyramids: Vec<&RasterPyramid> = pyramids.into_iter().collect();
    let ids = (0..pyramids.len())
        .map(|i| {
            u32::try_from(i)
                .ok()
                .and_then(|i| RasterId::new(i).ok())
                .ok_or(RasterWriteError::TooMany(pyramids.len()))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let placements = pyramids
        .iter()
        .map(|p| placement(p))
        .collect::<Result<Vec<_>, _>>()?;
    writer
        .rasters(&placements)
        .map_err(RasterWriteError::Pvec)?;

    for (id, pyramid) in ids.iter().zip(&pyramids) {
        let tiles = pyramid
            .tiles()
            .map(|(level, col, row, bytes)| TileIn {
                level,
                col,
                row,
                bytes: bytes.to_vec(),
            })
            .collect();
        writer
            .raster_tiles(*id, tiles)
            .map_err(RasterWriteError::Pvec)?;
    }
    Ok(ids)
}

/// The placement PVec stores for a pyramid: a conversion between the producer's fields and the
/// facade's.
pub(crate) fn placement(p: &RasterPyramid) -> Result<Placement, RasterWriteError> {
    Ok(Placement {
        pixel_to_source: p.pixel_to_source,
        width: p.width,
        height: p.height,
        visible_box: p.visible_box,
        tile_edge: u16::try_from(p.tile_edge).map_err(|_| RasterWriteError::Shape("tile edge"))?,
        levels: p.levels,
        tile_codec: match p.tile_codec {
            RasterCodec::Png => TileCodec::Png,
            RasterCodec::Jpeg => TileCodec::Jpeg,
            RasterCodec::Webp => TileCodec::Webp,
        },
        pixel_format: match p.channels {
            RasterChannels::Gray => PixelFormat::Gray,
            RasterChannels::Rgb => PixelFormat::Rgb,
            RasterChannels::Rgba => PixelFormat::Rgba,
        },
        bit_depth: p.bit_depth,
        opacity: 255,
        band: match p.band {
            RasterBand::Underlay => Band::Underlay,
            RasterBand::Overlay => Band::Overlay,
        },
        layer_index: None,
        source_digest: Some(p.source_digest),
        attribution: p.attribution.clone().unwrap_or_default(),
    })
}
