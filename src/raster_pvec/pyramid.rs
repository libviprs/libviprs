//! A raster pyramid, cut and encoded for a PVec raster stream (issue #1328, R2.4).
//!
//! ADR 0018 (libviprs/pvec) lets PVec store a raster as the encoded tiles of a pyramid, untouched,
//! plus a placement that maps pixels into the drawing's source space. This module is the producer
//! side that needs nothing from PVec: it plans, cuts and encodes the pyramid with the rules the
//! ADR puts on a producer, and hands back the tile bytes with their `(level, col, row)` and the
//! placement fields. The adapter that gives them to PVec's writer is thin on purpose, so that
//! everything the producer decides is testable without a container.
//!
//! The rules (ADR 0018, "Producers", and `docs/RASTER.md` section 1.2 in libviprs/pvec):
//!
//! * the plan is [`Layout::Google`] with **centring off and overlap 0**, whatever the caller's
//!   plan said (reported when it said otherwise). Google tiles are full squares, so an edge tile
//!   is padded with the background and the renderer clips to the visible box;
//! * `tile_size` is 256, 512 or 1024, and anything else is refused;
//! * the Google grid is a power of two on a side, so the tiles beyond `cols` by `rows` of a level
//!   (`cols = ceil(width_L / tile_size)`) are pure padding and are dropped;
//! * `skip_blanks` maps to a tile that is **not stored**. [`BlankTileStrategy::Placeholder`]
//!   writes a one-byte marker that is not an image, and is refused;
//! * the codec comes from the tile format (`Raw` is refused) and the pixel format is gray, RGB or
//!   RGBA at eight bits. A 16-bit source is converted to eight bits first, and a JPEG tile
//!   flattens alpha onto the background; each conversion is named in the report;
//! * the placement is `[a b c d tx ty]` in PVec's `Transform2` order with `-0.0` normalised, either
//!   given outright or composed from a stored `GeoTransform` and a calibration
//!   ([`placement`](super::placement)), and `source_digest` is the SHA-256 of the
//!   source samples.
//!
//! Nothing is resampled to fit a placement, and a lossless tile decodes to exactly the pixels the
//! producer cut (the producer-side claim of the ADR, tested here).

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use sha2::{Digest, Sha256};
use thiserror::Error;

use super::placement::{PlacementError, SourceCalibration, pixel_to_source};
use super::tiles::{RasterTileError, check_tile_edge, check_tile_len, tile_tags};
use crate::engine::{BlankTileStrategy, EngineConfig, EngineError};
use crate::geo::{GeoTransform, PixelOrigin};
use crate::pixel::PixelFormat;
use crate::planner::{Layout, PlannerError, PyramidPlan, PyramidPlanner};
use crate::raster::Raster;
use crate::sink::{SinkError, Tile, TileFormat, TileSink};
use crate::{EngineBuilder, EngineKind};

/// The longest attribution, in bytes (RASTER-07).
const MAX_ATTRIBUTION: usize = 255;

/// How the tiles are encoded: the codecs PVec stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RasterCodec {
    /// PNG.
    Png,
    /// JPEG, gray or RGB only.
    Jpeg,
    /// Lossless WebP.
    Webp,
}

/// The channels of a decoded tile, at eight bits. libviprs has no gray-alpha tile format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RasterChannels {
    /// One channel.
    Gray,
    /// Red, green, blue.
    Rgb,
    /// Red, green, blue and alpha.
    Rgba,
}

/// Where the placement comes from.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum RasterPlacement {
    /// `[a b c d tx ty]` in PVec's `Transform2` order, pixel origin at the top-left corner.
    Explicit([f64; 6]),
    /// Composed from the raster's own transform and the drawing's calibration.
    Composed {
        /// The raster's pixel to native transform, as the PMTiles archive stores it.
        pixel_to_native: GeoTransform,
        /// Where that transform puts the pixel origin.
        pixel_origin: PixelOrigin,
        /// The CRS the raster is in.
        raster_crs: String,
        /// The drawing's calibration.
        calibration: SourceCalibration,
    },
}

/// What a caller chooses about the pyramid. The tile edge comes from the requested plan.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RasterPyramidOptions {
    /// How tiles are encoded. [`TileFormat::Raw`] is refused.
    pub tile_format: TileFormat,
    /// Leave blank tiles out (a not-stored tile) rather than store them.
    pub skip_blanks: bool,
    /// Must stay [`BlankTileStrategy::Emit`]; the placeholder marker is not a tile.
    pub blank_strategy: BlankTileStrategy,
    /// Plain text of at most 255 bytes without control characters.
    pub attribution: Option<String>,
    /// The map from pixels into source space.
    pub placement: RasterPlacement,
}

impl RasterPyramidOptions {
    /// PNG tiles, blanks stored, no attribution, for `placement`.
    #[must_use]
    pub fn new(placement: RasterPlacement) -> Self {
        Self {
            tile_format: TileFormat::Png,
            skip_blanks: false,
            blank_strategy: BlankTileStrategy::Emit,
            attribution: None,
            placement,
        }
    }
}

/// Why a pyramid could not be produced.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum RasterPyramidError {
    /// A tile PVec would refuse: LIMIT-15 on its stored length, or a row the legal table lacks.
    #[error("a tile PVec would refuse")]
    Tile(#[source] RasterTileError),
    /// The tile edge is not one PVec accepts.
    #[error("tile size {0} is not 256, 512 or 1024")]
    TileSize(u32),
    /// The placeholder strategy writes a marker byte that is not an image.
    #[error("BlankTileStrategy::Placeholder writes a marker that is not a tile; use skip_blanks")]
    PlaceholderBlanks,
    /// PVec stores encoded tiles, and raw samples are not an encoding.
    #[error("raw tiles cannot be stored in a PVec raster; choose PNG, JPEG or WebP")]
    RawTiles,
    /// The source has a pixel format no PVec raster can carry.
    #[error("a raster of format {0:?} cannot be written as tiles; convert it to gray, RGB or RGBA")]
    PixelFormat(PixelFormat),
    /// The attribution breaks RASTER-07.
    #[error("attribution refused: {0}")]
    Attribution(&'static str),
    /// The placement could not be composed.
    #[error("the raster placement could not be composed")]
    Placement(#[source] PlacementError),
    /// A coefficient of an explicit placement was not finite.
    #[error("the explicit placement has a non-finite coefficient")]
    NonFinitePlacement,
    /// The requested plan could not be rebuilt.
    #[error("the pyramid plan could not be built")]
    Plan(#[source] PlannerError),
    /// The engine failed.
    #[error("cutting the pyramid failed")]
    Engine(#[source] EngineError),
}

/// What the producer did that the caller did not literally ask for.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct RasterPyramidReport {
    /// Pixel conversions, each named (`converted-bit-depth: Gray16 -> Gray8`).
    pub conversions: Vec<String>,
    /// Settings of the requested plan that were forced (`centre off`, `overlap 0`, `layout Google`).
    pub forced: Vec<String>,
    /// Tiles of the Google grid beyond a level's `cols` by `rows`, dropped as pure padding.
    pub dropped_padding_tiles: u64,
    /// Tiles inside the grid that were not stored because `skip_blanks` left them out.
    pub not_stored_tiles: u64,
}

/// A cut pyramid and its placement fields.
#[derive(Debug)]
#[non_exhaustive]
pub struct RasterPyramid {
    /// Pixels across the full resolution level.
    pub width: u32,
    /// Pixels down the full resolution level.
    pub height: u32,
    /// 256, 512 or 1024.
    pub tile_edge: u32,
    /// The number of levels; the last is full resolution.
    pub levels: u8,
    /// How the tiles are encoded.
    pub tile_codec: RasterCodec,
    /// The channels of a decoded tile.
    pub channels: RasterChannels,
    /// Bits per channel. Always 8 from this producer.
    pub bit_depth: u8,
    /// `[a b c d tx ty]`, `-0.0` normalised.
    pub pixel_to_source: [f64; 6],
    /// `[x0 y0 x1 y1]`: the whole image, since nothing is clipped here.
    pub visible_box: [u32; 4],
    /// SHA-256 of the source samples as the producer received them.
    pub source_digest: [u8; 32],
    /// The attribution, when given.
    pub attribution: Option<String>,
    /// The plan the pyramid was cut with: Google, centring off, overlap 0.
    pub plan: PyramidPlan,
    /// What the producer did on the way.
    pub report: RasterPyramidReport,
    tiles: BTreeMap<(u32, u32, u32), Vec<u8>>,
}

impl RasterPyramid {
    /// The encoded bytes of a tile, or `None` when it is not stored (left out as blank, or
    /// outside the level's grid).
    #[must_use]
    pub fn tile(&self, level: u32, col: u32, row: u32) -> Option<&[u8]> {
        self.tiles.get(&(level, col, row)).map(Vec::as_slice)
    }

    /// Every stored tile as `(level, col, row, bytes)`, by level, then column, then row.
    pub fn tiles(&self) -> impl Iterator<Item = (u32, u32, u32, &[u8])> {
        self.tiles
            .iter()
            .map(|(&(l, c, r), bytes)| (l, c, r, bytes.as_slice()))
    }

    /// `(cols, rows)` of a level: `ceil(width_L / tile_edge)` and the same for the height.
    #[must_use]
    pub fn grid(&self, level: u32) -> Option<(u32, u32)> {
        let l = self.plan.levels.get(level as usize)?;
        Some((
            l.width.div_ceil(self.tile_edge),
            l.height.div_ceil(self.tile_edge),
        ))
    }
}

/// Encode one tile of the engine's Google grid for PVec, or `None` when it lies beyond its level's
/// `cols` by `rows` and is pure padding. Shared by the collecting path and the streaming sink.
pub(crate) fn cut_tile(
    plan: &PyramidPlan,
    format: TileFormat,
    config: &Mutex<Option<EngineConfig>>,
    tile: &Tile,
) -> Result<Option<Vec<u8>>, SinkError> {
    let c = tile.coord;
    let level = plan
        .levels
        .get(c.level as usize)
        .ok_or(SinkError::InvalidCoord { coord: c })?;
    // The Google grid is a power of two on a side; past the image it is only padding.
    if c.col >= level.width.div_ceil(plan.tile_size) || c.row >= level.height.div_ceil(plan.tile_size)
    {
        return Ok(None);
    }
    Ok(Some(match format {
        TileFormat::Png => crate::sink::encode_png(&tile.raster)?,
        TileFormat::Jpeg { quality } => {
            let background = crate::sink::background_from(config);
            crate::sink::encode_jpeg(&tile.raster, quality, background)?
        }
        TileFormat::Webp => crate::sink::encode_webp(&tile.raster)?,
        TileFormat::Raw => return Err(SinkError::Unsupported("raw tiles".to_string())),
    }))
}

/// The sink the engine cuts into.
struct Collector {
    plan: PyramidPlan,
    format: TileFormat,
    config: Mutex<Option<EngineConfig>>,
    tiles: Mutex<BTreeMap<(u32, u32, u32), Vec<u8>>>,
    dropped: AtomicU64,
}

impl TileSink for Collector {
    fn write_tile(&self, tile: &Tile) -> Result<(), SinkError> {
        match cut_tile(&self.plan, self.format, &self.config, tile)? {
            Some(bytes) => {
                let c = tile.coord;
                crate::poison::recover(&self.tiles).insert((c.level, c.col, c.row), bytes);
            }
            None => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
        Ok(())
    }

    fn record_engine_config(&self, config: &EngineConfig) {
        *crate::poison::recover(&self.config) = Some(config.clone());
    }
}

/// A raster ready to be cut: the pyramid's placement fields and plan (no tiles yet), and the
/// eight-bit samples the tiles will be cut from.
#[derive(Debug)]
pub struct PreparedRaster<'a> {
    /// The placement fields, the plan and the report so far. [`RasterPyramid::tiles`] is empty.
    pub pyramid: RasterPyramid,
    /// The samples to cut: the caller's raster when it was already eight-bit, a converted copy
    /// otherwise.
    pub source: std::borrow::Cow<'a, Raster>,
}

/// Decide everything about the pyramid that does not need a tile: the checks, the placement, the
/// plan, the conversions. [`produce_raster_pyramid`] cuts the tiles into memory from the result,
/// and [`PvecSink`](crate::sink_pvec::PvecSink) streams them into a PVec writer.
///
/// `requested` supplies the tile edge and the source size (they must match `raster`); its
/// layout, centring and overlap are replaced as described in the module docs and the
/// replacements are listed in [`RasterPyramidReport::forced`].
///
/// # Errors
///
/// A [`RasterPyramidError`] names the rule: a tile edge outside 256, 512 and 1024, the placeholder
/// blank strategy, raw tiles, a pixel format with no tile encoding, a bad attribution, or a
/// placement that cannot be composed (the [`PlacementError`] is the `source()`).
pub fn prepare_raster<'a>(
    raster: &'a Raster,
    requested: &PyramidPlan,
    options: &RasterPyramidOptions,
) -> Result<PreparedRaster<'a>, RasterPyramidError> {
    let edge = requested.tile_size;
    check_tile_edge(edge).map_err(|_| RasterPyramidError::TileSize(edge))?;
    if matches!(
        options.blank_strategy,
        BlankTileStrategy::Placeholder | BlankTileStrategy::PlaceholderWithTolerance { .. }
    ) {
        return Err(RasterPyramidError::PlaceholderBlanks);
    }
    let codec = match options.tile_format {
        TileFormat::Png => RasterCodec::Png,
        TileFormat::Jpeg { .. } => RasterCodec::Jpeg,
        TileFormat::Webp => RasterCodec::Webp,
        TileFormat::Raw => return Err(RasterPyramidError::RawTiles),
    };
    if let Some(text) = &options.attribution {
        check_attribution(text)?;
    }
    let pixel_to_source = match &options.placement {
        RasterPlacement::Explicit(m) => {
            if !m.iter().all(|c| c.is_finite()) {
                return Err(RasterPyramidError::NonFinitePlacement);
            }
            m.map(|v| if v == 0.0 { 0.0 } else { v })
        }
        RasterPlacement::Composed {
            pixel_to_native,
            pixel_origin,
            raster_crs,
            calibration,
        } => pixel_to_source(pixel_to_native, *pixel_origin, raster_crs, calibration)
            .map_err(RasterPyramidError::Placement)?,
    };

    let mut report = RasterPyramidReport::default();
    let (source, channels) = eight_bit(raster, codec, &mut report)?;
    // The pair must be a row of the legal table. A JPEG tile of an RGBA source was flattened, so
    // it is looked up as RGB.
    let decoded = match channels {
        RasterChannels::Gray => PixelFormat::Gray8,
        RasterChannels::Rgb => PixelFormat::Rgb8,
        RasterChannels::Rgba => PixelFormat::Rgba8,
    };
    let (codec, channels) =
        tile_tags(options.tile_format, decoded, 8).map_err(RasterPyramidError::Tile)?;

    let (width, height) = (raster.width(), raster.height());
    let plan = PyramidPlanner::new(width, height, edge, 0, Layout::Google)
        .map_err(RasterPyramidError::Plan)?
        .plan();
    if requested.layout != Layout::Google {
        report
            .forced
            .push(format!("layout {:?} -> Google", requested.layout));
    }
    if requested.centre {
        report.forced.push("centre off".to_string());
    }
    if requested.overlap != 0 {
        report
            .forced
            .push(format!("overlap {} -> 0", requested.overlap));
    }

    let levels = u8::try_from(plan.levels.len()).expect("a u32 image has at most 32 levels");
    Ok(PreparedRaster {
        pyramid: RasterPyramid {
            width,
            height,
            tile_edge: edge,
            levels,
            tile_codec: codec,
            channels,
            bit_depth: 8,
            pixel_to_source,
            visible_box: [0, 0, width, height],
            source_digest: Sha256::digest(raster.data()).into(),
            attribution: options.attribution.clone(),
            plan,
            report,
            tiles: BTreeMap::new(),
        },
        source,
    })
}

/// Cut `raster` into the pyramid PVec stores, in memory. For a pyramid too large to hold, stream
/// it into a PVec writer with [`PvecSink`](crate::sink_pvec::PvecSink) instead.
///
/// `requested` and the errors are as for [`prepare_raster`], plus the engine's.
pub fn produce_raster_pyramid(
    raster: &Raster,
    requested: &PyramidPlan,
    options: &RasterPyramidOptions,
) -> Result<RasterPyramid, RasterPyramidError> {
    let PreparedRaster {
        mut pyramid,
        source,
    } = prepare_raster(raster, requested, options)?;
    let (plan, edge) = (pyramid.plan.clone(), pyramid.tile_edge);
    let collector = Collector {
        plan: plan.clone(),
        format: options.tile_format,
        config: Mutex::new(None),
        tiles: Mutex::new(BTreeMap::new()),
        dropped: AtomicU64::new(0),
    };
    // The strategy stays at its default, `Emit`, which was checked above.
    let config = EngineConfig::default().skip_blanks(options.skip_blanks);
    EngineBuilder::new(source.as_ref(), plan.clone(), &collector)
        .with_engine(EngineKind::Monolithic)
        .with_config(config)
        .run()
        .map_err(RasterPyramidError::Engine)?;

    let tiles = crate::poison::recover(&collector.tiles).clone();
    // LIMIT-15, before anything reaches PVec's writer.
    for bytes in tiles.values() {
        check_tile_len(edge, bytes.len()).map_err(RasterPyramidError::Tile)?;
    }
    let in_grid: u64 = plan
        .levels
        .iter()
        .map(|l| u64::from(l.width.div_ceil(edge)) * u64::from(l.height.div_ceil(edge)))
        .sum();

    pyramid.report.dropped_padding_tiles = collector.dropped.load(Ordering::Relaxed);
    pyramid.report.not_stored_tiles = in_grid - tiles.len() as u64;
    pyramid.tiles = tiles;
    Ok(pyramid)
}

/// RASTER-07: plain text, at most 255 bytes, no C0 or C1 controls, no U+2028 or U+2029 and no
/// bidirectional overrides.
fn check_attribution(text: &str) -> Result<(), RasterPyramidError> {
    if text.len() > MAX_ATTRIBUTION {
        return Err(RasterPyramidError::Attribution("longer than 255 bytes"));
    }
    let bad = |c: char| {
        c.is_control()
            || matches!(c, '\u{2028}' | '\u{2029}' | '\u{202A}'..='\u{202E}' | '\u{2066}'..='\u{2069}')
    };
    if text.chars().any(bad) {
        return Err(RasterPyramidError::Attribution(
            "a control character, line separator or bidirectional override",
        ));
    }
    Ok(())
}

/// The raster as eight-bit samples in a layout a tile encoder takes, with what was done named.
fn eight_bit<'a>(
    raster: &'a Raster,
    codec: RasterCodec,
    report: &mut RasterPyramidReport,
) -> Result<(std::borrow::Cow<'a, Raster>, RasterChannels), RasterPyramidError> {
    use std::borrow::Cow;
    let format = raster.format();
    let (target, channels) = match format {
        PixelFormat::Gray8 => (PixelFormat::Gray8, RasterChannels::Gray),
        PixelFormat::Gray16 => (PixelFormat::Gray8, RasterChannels::Gray),
        PixelFormat::Rgb8 => (PixelFormat::Rgb8, RasterChannels::Rgb),
        PixelFormat::Rgb16 => (PixelFormat::Rgb8, RasterChannels::Rgb),
        PixelFormat::Rgba8 => (PixelFormat::Rgba8, RasterChannels::Rgba),
        PixelFormat::Rgba16 => (PixelFormat::Rgba8, RasterChannels::Rgba),
        other => return Err(RasterPyramidError::PixelFormat(other)),
    };
    let source: Cow<'a, Raster> = if target == format {
        Cow::Borrowed(raster)
    } else {
        // Sixteen bits to eight, rounded: v * 255 / 65535, not the clipping cast.
        let data: Vec<u8> = raster
            .data()
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| {
                let v = u32::from(u16::from_ne_bytes([b[0], b[1]]));
                ((v * 255 + 32_767) / 65_535) as u8
            })
            .collect();
        report
            .conversions
            .push(format!("converted-bit-depth: {format:?} -> {target:?}"));
        let mut converted = Raster::new(raster.width(), raster.height(), target, data)
            .map_err(|_| RasterPyramidError::PixelFormat(format))?;
        converted.carry_meta_from(raster);
        Cow::Owned(converted)
    };
    // A JPEG tile has no alpha: the encoder flattens it onto the background, so the decoded
    // tile is RGB and the placement must say so (RASTER-04).
    if codec == RasterCodec::Jpeg && channels == RasterChannels::Rgba {
        report
            .conversions
            .push("Rgba8 -> Rgb8 (alpha flattened onto the background for JPEG)".to_string());
        return Ok((source, RasterChannels::Rgb));
    }
    Ok((source, channels))
}
