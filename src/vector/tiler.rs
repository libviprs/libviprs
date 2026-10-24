//! Turns world-space primitives into MVT tiles.
//!
//! ```text
//! Primitive[] -> bounds -> WorldTransform -> per zoom: tessellate -> clip -> quantise -> Tile
//! ```
//!
//! # The synthetic world
//!
//! A drawing has no geography. [`WorldTransform`] fits the drawing's bounds
//! into the unit square that zoom 0 is, centred, with the **same scale on both
//! axes** and a margin, and flips y because tiles run downwards and CAD runs
//! upwards. The aspect ratio therefore survives exactly, whatever the drawing's
//! shape. MapLibre sees an ordinary Web Mercator tileset; the archive header
//! carries the bounds the content occupies in degrees, and the metadata carries
//! the transform ([`TILER_METADATA_KEY`]) so a tool can map a tile back to
//! drawing units.
//!
//! # The precision budget
//!
//! "Vector sharp" is a promise over a range, and this is the arithmetic that
//! makes the range a number. At zoom `z` one tile unit is
//! `side / (2^z * extent)` drawing units. Three errors add up, each in tile
//! units and each a configuration value or a constant:
//!
//! | source | bound |
//! | --- | --- |
//! | quantising a vertex to the integer grid | 0.5 per axis, always |
//! | tessellating a curve ([`TilerConfig::deviation_units`]) | 0.25 by default |
//! | simplifying a run ([`TilerConfig::simplify_units`]) | 0.5 by default |
//!
//! so by default a point is at most 1.25 tile units from where the source put
//! it. With the default 4096 extent and 512 px tiles a unit is 1/8 px, which is
//! 0.16 px in the worst case and at most 1/16 px for quantisation alone.
//! [`TilerConfig::precision`] returns the bounds in drawing units at any zoom
//! so a caller can compare them to the smallest feature that matters.
//!
//! Measured on native x86_64 (`tests/vector_1032_tiler.rs`), a drawing a million
//! units from the origin, 1000 x 600 across, extent 4096:
//!
//! | zoom | one tile unit | worst vertex error | in tile units |
//! | --- | --- | --- | --- |
//! | 6 | 3.97e-3 | 2.17e-3 | 0.55 |
//! | 10 | 2.48e-4 | 1.05e-4 | 0.42 |
//! | 14 | 1.55e-5 | 8.50e-6 | 0.55 |
//! | 18 | 9.69e-7 | 6.58e-7 | 0.68 |
//! | 22 | 6.05e-8 | 3.18e-8 | 0.53 |
//!
//! and a circle of radius 250 deviates 0.73 to 0.94 tile units from the true
//! circle at zooms 2 to 8, against a budget of 1.46. **Zoom 22 is the deepest
//! zoom the tests cover**; [`MAX_SUPPORTED_ZOOM`] (24) is the ceiling the code
//! refuses to pass.
//!
//! MapLibre draws zooms above the archive's `max_zoom` by scaling its tiles, so
//! beyond `max_zoom` the grid grows with the picture: at `max_zoom + 3` a unit
//! is one pixel. The drawing is crisp through the generated range and for about
//! two further zooms of over-zoom, which is the range the tests assert. Pick
//! `max_zoom` so a unit at that zoom is below the smallest detail worth seeing:
//! [`suggest_max_zoom`] does it from a detail size.
//!
//! # What is bounded
//!
//! * **points per tile** ([`TilerConfig::max_points_per_tile`]): the longest
//!   runs are kept and the rest are dropped and counted;
//! * **text per tile** ([`TilerConfig::max_text_per_tile`]), tallest kept;
//! * **points held for one zoom level** ([`TilerConfig::max_level_points`]):
//!   exceeding it is an error rather than an out-of-memory;
//! * **bytes held for one zoom level** ([`TilerConfig::max_level_bytes`]):
//!   an estimate that counts the bookkeeping around every run and tile, which
//!   the point budget does not, so a line split over millions of tiles stops
//!   at the budget and not at six gigabytes;
//! * **one label** ([`TilerConfig::max_text_bytes`]): cut at a character
//!   boundary, counted and reported;
//! * **time** ([`TilerConfig::deadline`] and [`TilerConfig::cancel`]): polled
//!   per primitive, per split of a primitive over its tiles and per tile, so a
//!   single enormous primitive is stopped inside the primitive. A decoder
//!   behind [`build_archive_from_view`] cannot be interrupted from here, and is
//!   stopped at its first primitive after the limit; a decoder stuck inside
//!   native code is the worker process's to kill (see the performance report);
//! * **points per curve** ([`crate::vector::tessellate::MAX_POINTS_PER_CURVE`]):
//!   a curve past it is drawn at the cap and counted;
//! * **stored bytes per tile** ([`TilerConfig::max_tile_bytes`]): an oversized
//!   tile is detected and either reported or refused ([`OversizePolicy`]).
//!
//! Nothing is dropped silently: every cull, cap and failure is a counter in
//! [`TilerReport`].
//!
//! # Determinism
//!
//! Primitives are processed in input order, tiles are written in `(z, x, y)`
//! order from ordered maps, and nothing depends on hash iteration. The same
//! input and configuration give the same archive, byte for byte.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::{Duration, Instant};

use serde_json::json;
use thiserror::Error;

use crate::cad::{
    CadDrawing, CadError, DecodeReport, Diagnostic, DiagnosticCode, DisplayText, Primitive,
    PrimitiveSink, ViewKind, display_text,
};
use crate::cancel::CancelToken;
use crate::vector::archive::{
    FieldType, LayerSchema, VectorArchiveOptions, VectorArchiveWriter, VectorError,
};
use crate::vector::clip::{clip_slab, simplify};
use crate::vector::mvt::{Feature, Geometry, Layer, Point, Tile, Value};
use crate::vector::style::glyph_available;
use crate::vector::tessellate;

/// The metadata key the transform is stored under.
pub const TILER_METADATA_KEY: &str = "viprs_cad";

/// The deepest zoom the tiler will cut. `f64` carries a tile coordinate to
/// better than a thousandth of a unit at this depth with room to spare.
pub const MAX_SUPPORTED_ZOOM: u8 = 24;

/// What to do with a tile whose stored size passes the budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OversizePolicy {
    /// Store it and list it in the report.
    Report,
    /// Stop with [`TilerError::Oversized`].
    Error,
}

/// Everything configurable about a tiling run.
#[derive(Debug, Clone, PartialEq)]
pub struct TilerConfig {
    /// The MVT extent: tile coordinates run `0..extent`.
    pub extent: u32,
    /// Units of overlap each tile keeps past its edge, so a stroke and its
    /// joins are drawn the same on both sides of a boundary.
    pub buffer: u32,
    /// The deepest zoom generated. Zoom 0 is one tile holding the drawing.
    pub max_zoom: u8,
    /// Empty space left around the drawing, as a fraction of its larger side
    /// on each side.
    pub margin: f64,
    /// Chord deviation allowed when a curve is tessellated, in tile units.
    pub deviation_units: f64,
    /// Distance a dropped vertex may be from the kept run, in tile units.
    pub simplify_units: f64,
    /// A primitive whose extent is below this in both directions is not drawn
    /// at that zoom, in tile units.
    pub cull_units: f64,
    /// Text shorter than this is not emitted at that zoom, in tile units.
    pub min_text_units: f64,
    /// The most points one tile may carry.
    pub max_points_per_tile: usize,
    /// The most text features one tile may carry.
    pub max_text_per_tile: usize,
    /// The most points held while one zoom level is assembled.
    pub max_level_points: usize,
    /// The most stored (gzipped) bytes one tile should take.
    pub max_tile_bytes: usize,
    /// The most primitives held from one decode. A drawing of a block inserted
    /// a million times decodes to millions of primitives, and every one is
    /// held until the tiler has seen them all, so this is the memory bound for
    /// repeated blocks: about 150 bytes each, so the default is under a gigabyte.
    pub max_primitives: usize,
    /// What to do with a tile past `max_tile_bytes`.
    pub oversize: OversizePolicy,
    /// The most memory the tiles of one zoom level may take while they are
    /// assembled, as an estimate (see [`LEVEL_RUN_COST`] and
    /// [`LEVEL_TILE_COST`]). `max_level_points` counts points only, and a long
    /// line split over millions of tiles is millions of two-point runs whose
    /// bookkeeping costs far more than their points.
    pub max_level_bytes: usize,
    /// The longest label one text feature carries, in bytes. A longer one is
    /// cut at a character boundary and reported. The default, 16 KiB, is twice
    /// the longest label in the fixtures (a general-notes block of 8,192
    /// characters) and bounds a tile's decoded text at
    /// `max_text_per_tile * max_text_bytes`, which is 64 MiB.
    pub max_text_bytes: usize,
    /// Wall-clock budget for the whole run, from the moment it starts.
    pub deadline: Option<std::time::Duration>,
    /// Cooperative cancellation, polled as often as the deadline is.
    pub cancel: Option<crate::cancel::CancelToken>,
}

/// What one run of a level is estimated to cost beyond its points, in bytes.
pub const LEVEL_RUN_COST: usize = 64;
/// What one tile of a level is estimated to cost beyond its runs, in bytes.
pub const LEVEL_TILE_COST: usize = 160;

impl Default for TilerConfig {
    fn default() -> Self {
        Self {
            extent: 4096,
            buffer: 64,
            max_zoom: 12,
            margin: 0.02,
            deviation_units: 0.25,
            simplify_units: 0.5,
            cull_units: 0.5,
            min_text_units: 16.0,
            max_points_per_tile: 400_000,
            max_text_per_tile: 4_000,
            max_level_points: 64 << 20,
            max_tile_bytes: 500 << 10,
            max_primitives: 5_000_000,
            oversize: OversizePolicy::Report,
            max_level_bytes: 1 << 30,
            max_text_bytes: 16 << 10,
            deadline: None,
            cancel: None,
        }
    }
}

impl TilerConfig {
    /// Sets the deepest zoom.
    #[must_use]
    pub fn with_max_zoom(mut self, zoom: u8) -> Self {
        self.max_zoom = zoom;
        self
    }

    /// Sets the extent.
    #[must_use]
    pub fn with_extent(mut self, extent: u32) -> Self {
        self.extent = extent;
        self
    }

    /// Sets the wall-clock budget.
    #[must_use]
    pub fn with_deadline(mut self, deadline: std::time::Duration) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Sets the cancellation token.
    #[must_use]
    pub fn with_cancel(mut self, cancel: crate::cancel::CancelToken) -> Self {
        self.cancel = Some(cancel);
        self
    }

    /// Sets the byte budget for one zoom level.
    #[must_use]
    pub fn with_max_level_bytes(mut self, bytes: usize) -> Self {
        self.max_level_bytes = bytes;
        self
    }

    /// Sets the longest label, in bytes.
    #[must_use]
    pub fn with_max_text_bytes(mut self, bytes: usize) -> Self {
        self.max_text_bytes = bytes;
        self
    }

    fn validate(&self) -> Result<(), TilerError> {
        let bad = |m: &str| Err(TilerError::Config(m.to_owned()));
        if !(16..=crate::vector::mvt::MAX_EXTENT).contains(&self.extent) {
            return bad("extent must be in 16..=2^24");
        }
        if self.buffer >= self.extent {
            return bad("buffer must be smaller than the extent");
        }
        if self.max_zoom > MAX_SUPPORTED_ZOOM {
            return bad("max_zoom is above the deepest supported zoom");
        }
        for (name, v) in [
            ("margin", self.margin),
            ("deviation_units", self.deviation_units),
            ("simplify_units", self.simplify_units),
            ("cull_units", self.cull_units),
            ("min_text_units", self.min_text_units),
        ] {
            if !v.is_finite() || v < 0.0 {
                return Err(TilerError::Config(format!(
                    "{name} must be finite and not negative"
                )));
            }
        }
        if self.deviation_units <= 0.0 {
            return bad("deviation_units must be above zero");
        }
        if self.max_points_per_tile == 0
            || self.max_text_per_tile == 0
            || self.max_level_points == 0
        {
            return bad("the per-tile and per-level budgets must be above zero");
        }
        Ok(())
    }

    /// The errors this configuration allows, in drawing units, at `zoom`.
    #[must_use]
    pub fn precision(&self, transform: &WorldTransform, zoom: u8) -> PrecisionBudget {
        let unit = transform.source_units_per_tile_unit(zoom, self.extent);
        PrecisionBudget {
            zoom,
            tile_unit: unit,
            quantisation: 0.5 * unit,
            tessellation: self.deviation_units * unit,
            simplification: self.simplify_units * unit,
        }
    }
}

/// The error budget at one zoom, in drawing units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrecisionBudget {
    /// The zoom this is for.
    pub zoom: u8,
    /// One tile unit.
    pub tile_unit: f64,
    /// Rounding a vertex to the grid, per axis.
    pub quantisation: f64,
    /// Chord deviation of a tessellated curve.
    pub tessellation: f64,
    /// Distance a simplified-away vertex may be.
    pub simplification: f64,
}

impl PrecisionBudget {
    /// The worst a point can be from its source: the three errors, with the
    /// quantisation error counted along the diagonal.
    #[must_use]
    pub fn worst_case(&self) -> f64 {
        self.quantisation * std::f64::consts::SQRT_2 + self.tessellation + self.simplification
    }
}

/// The smallest `max_zoom` at which one tile unit is no larger than
/// `detail / 4` drawing units, for a world of the given side.
#[must_use]
pub fn suggest_max_zoom(side: f64, extent: u32, detail: f64) -> u8 {
    if !(side > 0.0 && detail > 0.0) {
        return 0;
    }
    let want = (side * 4.0 / (detail * f64::from(extent))).log2().ceil();
    want.clamp(0.0, f64::from(MAX_SUPPORTED_ZOOM)) as u8
}

/// Why a tiling run stopped.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum TilerError {
    /// A configuration value is out of range.
    #[error("invalid tiler configuration: {0}")]
    Config(String),
    /// No primitive produced any geometry, so there is nothing to fit.
    #[error("the view has no drawable geometry")]
    NoGeometry,
    /// The drawing's extent is not a finite box.
    #[error("the drawing's bounds are not finite")]
    NonFiniteBounds,
    /// One zoom level needed more points than it may hold.
    #[error("zoom {zoom} needs more than {limit} {unit} held at once")]
    LevelBudgetExceeded {
        /// The zoom.
        zoom: u8,
        /// The configured limit.
        limit: usize,
        /// What the limit counts: `"points"` or `"bytes"`.
        unit: &'static str,
    },
    /// A tile passed the byte budget under [`OversizePolicy::Error`].
    #[error("tile {z}/{x}/{y} is {bytes} bytes, past the {limit} byte budget")]
    Oversized {
        /// Zoom.
        z: u8,
        /// Column.
        x: u32,
        /// Row.
        y: u32,
        /// Stored size.
        bytes: usize,
        /// The budget.
        limit: usize,
    },
    /// The decoder failed before the view was complete. The report holds what
    /// it had said up to that point.
    #[error("decoding failed: {error}")]
    DecodeFailed {
        /// The failure.
        error: CadError,
        /// Everything the decode reported before it stopped.
        report: Box<DecodeReport>,
    },
    /// The decoder failed in a way that has no report: the view does not exist.
    #[error(transparent)]
    Decode(#[from] CadError),
    /// The view is a layout, and paper space is not enabled.
    ///
    /// Model Space is the production target. A layout needs its viewports
    /// transformed and clipped, and until those fixtures pass drawing one
    /// would put geometry on the sheet that the viewport hides.
    #[error(
        "view {index} ({name:?}) is a layout; only Model Space is enabled until viewport transform and clipping fixtures pass"
    )]
    ViewNotEnabled {
        /// The view's index.
        index: u32,
        /// Its name.
        name: String,
    },
    /// The decode produced more primitives than may be held.
    #[error(
        "the view holds more than {limit} primitives, which is the most that may be held at once"
    )]
    TooManyPrimitives {
        /// The configured limit.
        limit: usize,
    },
    /// The run was cancelled through [`TilerConfig::cancel`].
    #[error("the run was cancelled")]
    Cancelled,
    /// The run passed [`TilerConfig::deadline`].
    #[error("the run passed its {limit:?} deadline")]
    DeadlineExceeded {
        /// The configured budget.
        limit: std::time::Duration,
    },
    /// The archive writer refused something.
    #[error(transparent)]
    Archive(#[from] VectorError),
}

/// Drawing units to the unit square of zoom 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorldTransform {
    cx: f64,
    cy: f64,
    side: f64,
}

impl WorldTransform {
    /// Fits `bounds` (`[min_x, min_y, max_x, max_y]`) into the unit square,
    /// centred, with one scale on both axes and `margin` of empty space on
    /// each side.
    ///
    /// # Errors
    ///
    /// [`TilerError::NonFiniteBounds`] for a box with a non-finite or inverted
    /// side.
    pub fn from_bounds(bounds: [f64; 4], margin: f64) -> Result<Self, TilerError> {
        if bounds.iter().any(|v| !v.is_finite()) || bounds[2] < bounds[0] || bounds[3] < bounds[1] {
            return Err(TilerError::NonFiniteBounds);
        }
        let w = bounds[2] - bounds[0];
        let h = bounds[3] - bounds[1];
        let larger = w.max(h);
        // A drawing that is a single point still needs a world to sit in.
        let larger = if larger > 0.0 { larger } else { 1.0 };
        let side = larger * (1.0 + 2.0 * margin);
        if !side.is_finite() {
            return Err(TilerError::NonFiniteBounds);
        }
        Ok(Self {
            cx: (bounds[0] + bounds[2]) / 2.0,
            cy: (bounds[1] + bounds[3]) / 2.0,
            side,
        })
    }

    /// The side of the world square, in drawing units.
    #[must_use]
    pub fn side(&self) -> f64 {
        self.side
    }

    /// The drawing point at the middle of the world.
    #[must_use]
    pub fn centre(&self) -> [f64; 2] {
        [self.cx, self.cy]
    }

    /// A drawing point as `(u, v)` in the unit square, `v` increasing downward.
    ///
    /// The offset is taken from the centre before scaling, which is what keeps
    /// a drawing a million units from the origin as precise as one at it.
    #[must_use]
    pub fn unit(&self, p: [f64; 2]) -> [f64; 2] {
        [
            (p[0] - self.cx) / self.side + 0.5,
            0.5 - (p[1] - self.cy) / self.side,
        ]
    }

    /// The inverse of [`unit`](Self::unit).
    #[must_use]
    pub fn from_unit(&self, uv: [f64; 2]) -> [f64; 2] {
        [
            self.cx + (uv[0] - 0.5) * self.side,
            self.cy - (uv[1] - 0.5) * self.side,
        ]
    }

    /// Drawing units in one tile unit at `zoom`.
    #[must_use]
    pub fn source_units_per_tile_unit(&self, zoom: u8, extent: u32) -> f64 {
        self.side / (f64::from(1u32 << zoom.min(31)) * f64::from(extent))
    }

    /// The drawing point a tile-local coordinate stands for.
    #[must_use]
    pub fn tile_to_source(
        &self,
        zoom: u8,
        tx: u32,
        ty: u32,
        local: [f64; 2],
        extent: u32,
    ) -> [f64; 2] {
        let n = f64::from(1u32 << zoom.min(31));
        let e = f64::from(extent);
        self.from_unit([
            (f64::from(tx) + local[0] / e) / n,
            (f64::from(ty) + local[1] / e) / n,
        ])
    }

    /// A drawing point as longitude and latitude in degrees, for the archive
    /// header: the unit square is the Web Mercator world.
    #[must_use]
    pub fn lon_lat(&self, p: [f64; 2]) -> (f64, f64) {
        let [u, v] = self.unit(p);
        let lon = (u - 0.5) * 360.0;
        let lat = (std::f64::consts::PI * (1.0 - 2.0 * v))
            .sinh()
            .atan()
            .to_degrees();
        (lon, lat)
    }
}

/// Counters for everything the run did to the drawing. Nothing is dropped
/// without a number here.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct TilerReport {
    /// Primitives given to the tiler.
    pub primitives: u64,
    /// Primitives that were drawable at some zoom.
    pub drawable: u64,
    /// Tiles written.
    pub tiles: u64,
    /// Tiles written per zoom.
    pub tiles_per_zoom: Vec<u64>,
    /// Line points written, after clipping and simplification.
    pub points: u64,
    /// Text features written.
    pub text: u64,
    /// (primitive, zoom) pairs skipped for being smaller than `cull_units`.
    pub culled_subpixel: u64,
    /// (text, zoom) pairs skipped for being shorter than `min_text_units`.
    pub culled_text: u64,
    /// Runs that collapsed to one point when quantised.
    pub degenerate_after_quantisation: u64,
    /// Runs dropped to hold a tile to `max_points_per_tile`.
    pub dropped_for_point_budget: u64,
    /// Text dropped to hold a tile to `max_text_per_tile`.
    pub dropped_for_text_budget: u64,
    /// (curve, zoom) pairs stopped by the per-curve point cap.
    pub curves_capped: u64,
    /// Primitives the tessellator could not evaluate, so were not drawn.
    pub undrawable: u64,
    /// The most stored bytes any tile took.
    pub largest_tile_bytes: usize,
    /// Tiles past the byte budget, up to 64 of them, as `(z, x, y, bytes)`.
    pub oversized: Vec<(u8, u32, u32, usize)>,
    /// How many tiles were past the byte budget, including those not listed.
    pub oversized_count: u64,
    /// The error budget at `max_zoom`, in drawing units.
    pub precision_at_max_zoom: Option<PrecisionBudget>,
    /// The world the run used.
    pub world_side: f64,
    /// Text that was nothing once its formatting was read out of it.
    pub culled_empty_text: u64,
    /// Labels cut to [`TilerConfig::max_text_bytes`].
    pub text_truncated: u64,
    /// Fonts that text asked for, with how many strings asked. The output has
    /// one font, so each of these was substituted.
    pub fonts_substituted: BTreeMap<String, u64>,
    /// Characters the served glyph files cannot draw, with how many times they
    /// appear.
    pub glyphs_unavailable: BTreeMap<char, u64>,
    /// MTEXT formatting codes that were read and not honoured, by letter.
    pub text_codes_ignored: BTreeMap<char, u64>,
    /// One diagnostic per distinct thing worth saying (a font, a glyph, a
    /// formatting code), up to [`TilerReport::DIAGNOSTIC_LIMIT`].
    pub diagnostics: Vec<Diagnostic>,
    /// Diagnostics past the limit.
    pub dropped_diagnostics: u64,
}

impl TilerReport {
    /// The most diagnostics kept.
    pub const DIAGNOSTIC_LIMIT: usize = 256;

    fn say(&mut self, d: Diagnostic) {
        if self.diagnostics.len() < Self::DIAGNOSTIC_LIMIT {
            self.diagnostics.push(d);
        } else {
            self.dropped_diagnostics += 1;
        }
    }
}

/// Where finished tiles go.
pub trait TileOutput {
    /// Takes one tile. Returns its stored size when it was kept.
    ///
    /// # Errors
    ///
    /// Whatever the output refuses with.
    fn put(&mut self, z: u8, x: u32, y: u32, tile: &Tile) -> Result<Option<usize>, TilerError>;
}

impl<W: std::io::Write + std::io::Seek> TileOutput for VectorArchiveWriter<W> {
    fn put(&mut self, z: u8, x: u32, y: u32, tile: &Tile) -> Result<Option<usize>, TilerError> {
        Ok(self.add_tile_sized(z, x, y, tile)?)
    }
}

/// Keeps every tile in memory, for tests and small drawings.
#[derive(Debug, Default)]
pub struct CollectTiles {
    /// The tiles by `(z, x, y)`.
    pub tiles: BTreeMap<(u8, u32, u32), Tile>,
}

impl TileOutput for CollectTiles {
    fn put(&mut self, z: u8, x: u32, y: u32, tile: &Tile) -> Result<Option<usize>, TilerError> {
        if tile.is_empty() {
            return Ok(None);
        }
        let n = tile.encode().map_err(VectorError::from)?.len();
        self.tiles.insert((z, x, y), tile.clone());
        Ok(Some(n))
    }
}

// ---------------------------------------------------------------------------
// Stopping, and bounding one string
// ---------------------------------------------------------------------------

/// Polls the deadline and the cancellation token. Cheap enough to call per
/// primitive and per split: the token is one atomic load, and the clock is read
/// on one call in 32.
#[derive(Clone)]
struct Guard {
    start: Instant,
    limit: Option<Duration>,
    cancel: Option<CancelToken>,
    tick: u32,
}

impl Guard {
    fn new(config: &TilerConfig) -> Self {
        Self {
            start: Instant::now(),
            limit: config.deadline,
            cancel: config.cancel.clone(),
            tick: 0,
        }
    }

    fn check(&mut self) -> Result<(), TilerError> {
        if self.cancel.as_ref().is_some_and(CancelToken::is_cancelled) {
            return Err(TilerError::Cancelled);
        }
        self.tick = self.tick.wrapping_add(1);
        if let Some(limit) = self.limit
            && self.tick % 32 == 1
            && self.start.elapsed() >= limit
        {
            return Err(TilerError::DeadlineExceeded { limit });
        }
        Ok(())
    }
}

/// `s` cut to at most `max` bytes at a character boundary, and whether it was.
fn cut_at(s: &str, max: usize) -> (&str, bool) {
    if s.len() <= max {
        return (s, false);
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    (&s[..end], true)
}

/// What a label displays, held to `max` bytes, and whether it was cut.
///
/// The raw string is cut first, at four times the cap, so that a hostile
/// string is never read (or copied) whole: formatting codes sit in the raw
/// string and not in what is shown, which is why the raw allowance is larger.
fn shown(content: &str, max: usize) -> (DisplayText, bool) {
    let (raw, cut_raw) = cut_at(content, max.saturating_mul(4));
    let mut d = display_text(raw);
    let (kept, cut_text) = cut_at(&d.text, max);
    let keep = kept.len();
    d.text.truncate(keep);
    (d, cut_raw || cut_text)
}

// ---------------------------------------------------------------------------
// Flattening one primitive
// ---------------------------------------------------------------------------

enum Flat {
    Lines {
        runs: Vec<Vec<[f64; 2]>>,
        capped: bool,
    },
    Text {
        at: [f64; 2],
        height: f64,
        rotation: f64,
        text: String,
    },
    Undrawable,
}

fn flatten(p: &Primitive, tol: f64, max_text: usize) -> Flat {
    match p {
        Primitive::Line(l) => Flat::Lines {
            runs: vec![vec![
                [l.start().x(), l.start().y()],
                [l.end().x(), l.end().y()],
            ]],
            capped: false,
        },
        Primitive::Polyline(pl) => {
            let v: Vec<[f64; 3]> = pl.vertices().iter().map(|p| p.into_array()).collect();
            let t = tessellate::bulge_run(
                &v,
                pl.bulges(),
                pl.normal().into_array(),
                pl.is_closed(),
                tol,
            );
            Flat::Lines {
                runs: vec![t.points],
                capped: t.capped,
            }
        }
        Primitive::Polygon(pg) => {
            // A closed boundary, drawn as an outline until solid fills land.
            let v: Vec<[f64; 3]> = pg.vertices().iter().map(|p| p.into_array()).collect();
            let t = tessellate::bulge_run(&v, pg.bulges(), pg.normal().into_array(), true, tol);
            Flat::Lines {
                runs: vec![t.points],
                capped: t.capped,
            }
        }
        Primitive::Arc(a) => {
            let t = tessellate::arc(a, tol);
            Flat::Lines {
                runs: vec![t.points],
                capped: t.capped,
            }
        }
        Primitive::Circle(c) => {
            let t = tessellate::circle(c, tol);
            Flat::Lines {
                runs: vec![t.points],
                capped: t.capped,
            }
        }
        Primitive::Ellipse(e) => {
            let t = tessellate::ellipse(e, tol);
            Flat::Lines {
                runs: vec![t.points],
                capped: t.capped,
            }
        }
        Primitive::Spline(s) => match tessellate::spline(s, tol) {
            Some(t) => Flat::Lines {
                runs: vec![t.points],
                capped: t.capped,
            },
            None => Flat::Undrawable,
        },
        Primitive::Text(t) => Flat::Text {
            at: [t.position().x(), t.position().y()],
            height: t.height(),
            rotation: t.rotation(),
            // What a person reads: MTEXT formatting is read out of it here, once,
            // so bounds and tiles agree on the string.
            text: shown(t.content(), max_text).0.text,
        },
    }
}

fn bbox_of(runs: &[Vec<[f64; 2]>]) -> Option<[f64; 4]> {
    let mut b = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    let mut any = false;
    for p in runs.iter().flatten() {
        any = true;
        b[0] = b[0].min(p[0]);
        b[1] = b[1].min(p[1]);
        b[2] = b[2].max(p[0]);
        b[3] = b[3].max(p[1]);
    }
    (any && b.iter().all(|v| v.is_finite())).then_some(b)
}

/// The larger side of the box around some points, which is the scale a curve
/// through or near them is judged at.
fn hull_size(points: &[crate::cad::Point3]) -> f64 {
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for p in points {
        lo[0] = lo[0].min(p.x());
        lo[1] = lo[1].min(p.y());
        hi[0] = hi[0].max(p.x());
        hi[1] = hi[1].max(p.y());
    }
    let size = (hi[0] - lo[0]).max(hi[1] - lo[1]);
    if size.is_finite() && size > 0.0 {
        size
    } else {
        1.0
    }
}

/// A primitive's world box, from a tessellation fine enough for its own size.
fn primitive_bounds(p: &Primitive, max_text: usize) -> Option<[f64; 4]> {
    // First pass at a size-relative tolerance: radius for the round things,
    // a generous fraction of the control box for splines.
    let rough = match p {
        Primitive::Arc(a) => a.radius(),
        Primitive::Circle(c) => c.radius(),
        Primitive::Ellipse(e) => e.major_axis().length(),
        Primitive::Spline(s) => hull_size(s.controls()),
        Primitive::Polyline(pl) => hull_size(pl.vertices()),
        Primitive::Polygon(pg) => hull_size(pg.vertices()),
        _ => 1.0,
    };
    let tol = (rough * 1e-4).max(f64::MIN_POSITIVE);
    match flatten(p, tol, max_text) {
        Flat::Lines { runs, .. } => {
            let b = bbox_of(&runs)?;
            Some([b[0] - tol, b[1] - tol, b[2] + tol, b[3] + tol])
        }
        Flat::Text {
            at,
            height,
            rotation,
            text,
        } => {
            if text.trim().is_empty() {
                return None;
            }
            // Text has no measured width here: it is estimated from the glyph
            // count of its longest line, which is enough to keep it inside the
            // world, and the lines stack upward from the anchor.
            let longest = text.lines().map(|l| l.chars().count()).max().unwrap_or(0);
            let w = height * 0.6 * longest as f64;
            let height = height * text.lines().count().max(1) as f64;
            let (s, c) = rotation.sin_cos();
            let end = [at[0] + w * c, at[1] + w * s];
            Some([
                at[0].min(end[0]) - height,
                at[1].min(end[1]) - height,
                at[0].max(end[0]) + height,
                at[1].max(end[1]) + height,
            ])
        }
        Flat::Undrawable => None,
    }
}

// ---------------------------------------------------------------------------
// One zoom level
// ---------------------------------------------------------------------------

#[derive(Default)]
struct TileBuf {
    lines: Vec<Vec<Point>>,
    text: Vec<TextFeature>,
}

struct TextFeature {
    at: Point,
    text: String,
    height: f64,
    h0: f64,
    rotation: f64,
}

struct Level {
    zoom: u8,
    tiles: BTreeMap<(u32, u32), TileBuf>,
    points: usize,
    runs: usize,
}

impl Level {
    /// What this level is estimated to hold, in bytes: the points themselves
    /// and the bookkeeping around each run and tile.
    fn estimated_bytes(&self) -> usize {
        self.points
            .saturating_mul(std::mem::size_of::<Point>())
            .saturating_add(self.runs.saturating_mul(LEVEL_RUN_COST))
            .saturating_add(self.tiles.len().saturating_mul(LEVEL_TILE_COST))
    }
}

struct Ctx<'a> {
    config: &'a TilerConfig,
    report: &'a mut TilerReport,
    guard: &'a mut Guard,
}

/// Distributes already-flattened runs, in grid coordinates, over the tiles
/// they touch.
///
/// The grid range is halved along its longer side, the runs are clipped to
/// each half (plus the buffer) and the halves recursed into, so a long line
/// costs `points * log(grid)` rather than `points * tiles`. The stack is
/// explicit and its depth is at most `2 * zoom`.
fn distribute(
    ctx: &mut Ctx<'_>,
    level: &mut Level,
    runs: Vec<Vec<[f64; 2]>>,
) -> Result<(), TilerError> {
    let Some(b) = bbox_of(&runs) else {
        return Ok(());
    };
    let n = 1u64 << level.zoom;
    let max_ix = (n - 1) as f64;
    let buf = f64::from(ctx.config.buffer) / f64::from(ctx.config.extent);
    let lo = |v: f64| (v - buf).floor().clamp(0.0, max_ix) as u32;
    let hi = |v: f64| (v + buf).floor().clamp(0.0, max_ix) as u32;
    let (x0, x1, y0, y1) = (lo(b[0]), hi(b[2]), lo(b[1]), hi(b[3]));

    let mut stack = vec![(runs, x0, x1, y0, y1)];
    while let Some((runs, x0, x1, y0, y1)) = stack.pop() {
        ctx.guard.check()?;
        if x0 == x1 && y0 == y1 {
            emit_runs(ctx, level, x0, y0, &runs)?;
            continue;
        }
        let split_x = (x1 - x0) >= (y1 - y0);
        let (a0, a1) = if split_x { (x0, x1) } else { (y0, y1) };
        let mid = a0 + (a1 - a0) / 2;
        let axis = usize::from(!split_x);
        let halves = [(a0, mid), (mid + 1, a1)];
        for (h0, h1) in halves {
            let lo_edge = f64::from(h0) - buf;
            let hi_edge = f64::from(h1) + 1.0 + buf;
            let mut part: Vec<Vec<[f64; 2]>> = Vec::new();
            for run in &runs {
                part.extend(clip_slab(run, axis, lo_edge, hi_edge));
            }
            if part.is_empty() {
                continue;
            }
            if split_x {
                stack.push((part, h0, h1, y0, y1));
            } else {
                stack.push((part, x0, x1, h0, h1));
            }
        }
    }
    Ok(())
}

fn emit_runs(
    ctx: &mut Ctx<'_>,
    level: &mut Level,
    tx: u32,
    ty: u32,
    runs: &[Vec<[f64; 2]>],
) -> Result<(), TilerError> {
    let e = f64::from(ctx.config.extent);
    let (ox, oy) = (f64::from(tx), f64::from(ty));
    for run in runs {
        let local: Vec<[f64; 2]> = run
            .iter()
            .map(|p| [(p[0] - ox) * e, (p[1] - oy) * e])
            .collect();
        let simplified = simplify(&local, ctx.config.simplify_units);
        let mut pts: Vec<Point> = Vec::with_capacity(simplified.len());
        for p in &simplified {
            let q = [p[0].round() as i32, p[1].round() as i32];
            if pts.last() != Some(&q) {
                pts.push(q);
            }
        }
        if pts.len() < 2 {
            ctx.report.degenerate_after_quantisation += 1;
            continue;
        }
        level.points += pts.len();
        level.runs += 1;
        level.tiles.entry((tx, ty)).or_default().lines.push(pts);
        if level.points > ctx.config.max_level_points {
            return Err(TilerError::LevelBudgetExceeded {
                zoom: level.zoom,
                limit: ctx.config.max_level_points,
                unit: "points",
            });
        }
        if level.estimated_bytes() > ctx.config.max_level_bytes {
            return Err(TilerError::LevelBudgetExceeded {
                zoom: level.zoom,
                limit: ctx.config.max_level_bytes,
                unit: "bytes",
            });
        }
    }
    Ok(())
}

/// Reads what a text primitive's formatting and characters say about fidelity
/// and files it: fonts the output cannot honour, glyphs it cannot draw, codes
/// it dropped. Once per distinct thing as a diagnostic, always as a count.
fn note_text(report: &mut TilerReport, t: &crate::cad::Text, max_text: usize) {
    let (d, cut) = shown(t.content(), max_text);
    let handle = t.origin().item_handle();
    let with = |diag: Diagnostic| match handle {
        Some(h) => diag.with_entity(h),
        None => diag,
    };
    if cut {
        report.text_truncated += 1;
        report.say(with(Diagnostic::from_code(
            DiagnosticCode::TEXT_TRUNCATED,
            format!(
                "a label is longer than the {max_text} bytes one feature carries, so it is cut there"
            ),
        )));
    }
    for font in &d.fonts {
        let first = !report.fonts_substituted.contains_key(font);
        *report.fonts_substituted.entry(font.clone()).or_default() += 1;
        if first {
            report.say(with(Diagnostic::from_code(
                DiagnosticCode::FONT_SUBSTITUTED,
                format!(
                    "text asks for the font {font:?}, which the output does not carry; every label is drawn in Noto Sans Regular"
                ),
            )));
        }
    }
    for (code, n) in &d.ignored {
        let first = !report.text_codes_ignored.contains_key(code);
        *report.text_codes_ignored.entry(*code).or_default() += n;
        if first {
            report.say(with(Diagnostic::from_code(
                DiagnosticCode::TEXT_FORMAT_IGNORED,
                format!(
                    "the MTEXT code \\{code} is read and not honoured, so that formatting is lost"
                ),
            )));
        }
    }
    for c in d
        .text
        .chars()
        .filter(|c| !c.is_whitespace() && !glyph_available(*c))
    {
        let first = !report.glyphs_unavailable.contains_key(&c);
        *report.glyphs_unavailable.entry(c).or_default() += 1;
        if first {
            report.say(with(Diagnostic::from_code(
                DiagnosticCode::GLYPH_UNAVAILABLE,
                format!(
                    "U+{:04X} has no glyph in the font files the style serves",
                    u32::from(c)
                ),
            )));
        }
    }
}

/// Tiles `primitives` and hands every tile to `out`.
///
/// # Errors
///
/// [`TilerError`]. A run that fails part way has written the tiles of the
/// zooms it finished; an archive writer drops its staging files and publishes
/// nothing.
pub fn tile_primitives(
    primitives: &[Primitive],
    config: &TilerConfig,
    out: &mut dyn TileOutput,
) -> Result<(TilerReport, WorldTransform), TilerError> {
    config.validate()?;
    tile_guarded(primitives, config, out, &mut Guard::new(config))
}

fn tile_guarded(
    primitives: &[Primitive],
    config: &TilerConfig,
    out: &mut dyn TileOutput,
    guard: &mut Guard,
) -> Result<(TilerReport, WorldTransform), TilerError> {
    let mut report = TilerReport {
        primitives: primitives.len() as u64,
        ..TilerReport::default()
    };

    // Pass 1: each primitive's own box, and the union.
    let mut boxes: Vec<Option<[f64; 4]>> = Vec::with_capacity(primitives.len());
    let mut all = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for p in primitives {
        guard.check()?;
        if let Primitive::Text(t) = p {
            note_text(&mut report, t, config.max_text_bytes);
        }
        let b = primitive_bounds(p, config.max_text_bytes);
        match b {
            None if matches!(p, Primitive::Text(t) if shown(t.content(), config.max_text_bytes).0.is_blank()) =>
            {
                // Formatting and nothing else: there is nothing to draw, and
                // that is not a failure to draw it.
                report.culled_empty_text += 1;
                boxes.push(None);
            }
            Some(b) if b.iter().all(|v| v.is_finite()) => {
                all[0] = all[0].min(b[0]);
                all[1] = all[1].min(b[1]);
                all[2] = all[2].max(b[2]);
                all[3] = all[3].max(b[3]);
                report.drawable += 1;
                boxes.push(Some(b));
            }
            _ => {
                report.undrawable += 1;
                boxes.push(None);
            }
        }
    }
    if report.drawable == 0 {
        return Err(TilerError::NoGeometry);
    }
    let world = WorldTransform::from_bounds(all, config.margin)?;
    report.world_side = world.side();
    report.precision_at_max_zoom = Some(config.precision(&world, config.max_zoom));
    report.tiles_per_zoom = vec![0; usize::from(config.max_zoom) + 1];

    // Pass 2: a zoom at a time.
    for zoom in 0..=config.max_zoom {
        let scale = f64::from(1u32 << zoom);
        let unit = world.source_units_per_tile_unit(zoom, config.extent);
        let tol = config.deviation_units * unit;
        let mut level = Level {
            zoom,
            tiles: BTreeMap::new(),
            points: 0,
            runs: 0,
        };
        let mut ctx = Ctx {
            config,
            report: &mut report,
            guard: &mut *guard,
        };

        for (p, bx) in primitives.iter().zip(&boxes) {
            let Some(bx) = bx else { continue };
            ctx.guard.check()?;
            let is_text = matches!(p, Primitive::Text(_));
            if !is_text
                && (bx[2] - bx[0]) < config.cull_units * unit
                && (bx[3] - bx[1]) < config.cull_units * unit
            {
                ctx.report.culled_subpixel += 1;
                continue;
            }
            match flatten(p, tol, config.max_text_bytes) {
                Flat::Undrawable => ctx.report.undrawable += 1,
                Flat::Lines { runs, capped } => {
                    if capped {
                        ctx.report.curves_capped += 1;
                    }
                    let grid: Vec<Vec<[f64; 2]>> = runs
                        .iter()
                        .map(|r| {
                            r.iter()
                                .map(|q| {
                                    let uv = world.unit(*q);
                                    [uv[0] * scale, uv[1] * scale]
                                })
                                .collect()
                        })
                        .collect();
                    distribute(&mut ctx, &mut level, grid)?;
                }
                Flat::Text {
                    at,
                    height,
                    rotation,
                    text,
                } => {
                    let height_units = height / unit;
                    if height_units < config.min_text_units {
                        ctx.report.culled_text += 1;
                        continue;
                    }
                    let uv = world.unit(at);
                    let g = [uv[0] * scale, uv[1] * scale];
                    let max_ix = scale - 1.0;
                    let (tx, ty) = (g[0].floor(), g[1].floor());
                    if !(0.0..=max_ix).contains(&tx) || !(0.0..=max_ix).contains(&ty) {
                        continue;
                    }
                    let e = f64::from(config.extent);
                    let local = [
                        ((g[0] - tx) * e).round() as i32,
                        ((g[1] - ty) * e).round() as i32,
                    ];
                    // MapLibre rotates clockwise and the plan is drawn upright,
                    // so a counter-clockwise CAD angle is a negative one here.
                    let degrees = (-rotation.to_degrees()).rem_euclid(360.0);
                    level.points += 1;
                    level
                        .tiles
                        .entry((tx as u32, ty as u32))
                        .or_default()
                        .text
                        .push(TextFeature {
                            at: local,
                            text,
                            height,
                            h0: height / world.side() * 512.0,
                            rotation: degrees,
                        });
                }
            }
        }

        // Write the level.
        for ((tx, ty), buf) in std::mem::take(&mut level.tiles) {
            ctx.guard.check()?;
            let tile = build_tile(&mut ctx, buf);
            let Some(bytes) = out.put(zoom, tx, ty, &tile)? else {
                continue;
            };
            ctx.report.tiles += 1;
            ctx.report.tiles_per_zoom[usize::from(zoom)] += 1;
            ctx.report.largest_tile_bytes = ctx.report.largest_tile_bytes.max(bytes);
            if bytes > config.max_tile_bytes {
                ctx.report.oversized_count += 1;
                if ctx.report.oversized.len() < 64 {
                    ctx.report.oversized.push((zoom, tx, ty, bytes));
                }
                if config.oversize == OversizePolicy::Error {
                    return Err(TilerError::Oversized {
                        z: zoom,
                        x: tx,
                        y: ty,
                        bytes,
                        limit: config.max_tile_bytes,
                    });
                }
            }
        }
    }
    Ok((report, world))
}

fn build_tile(ctx: &mut Ctx<'_>, mut buf: TileBuf) -> Tile {
    let cfg = ctx.config;
    // Hold the tile to its point budget by dropping the shortest runs first.
    // The sort is stable and keyed on length, so the same input drops the same
    // runs.
    let total: usize = buf.lines.iter().map(Vec::len).sum();
    if total > cfg.max_points_per_tile {
        let mut order: Vec<usize> = (0..buf.lines.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(buf.lines[i].len()));
        let mut kept = 0usize;
        let mut keep = vec![false; buf.lines.len()];
        for i in order {
            if kept + buf.lines[i].len() > cfg.max_points_per_tile {
                continue;
            }
            kept += buf.lines[i].len();
            keep[i] = true;
        }
        let before = buf.lines.len();
        let mut it = keep.iter();
        buf.lines.retain(|_| *it.next().unwrap_or(&false));
        ctx.report.dropped_for_point_budget += (before - buf.lines.len()) as u64;
    }
    if buf.text.len() > cfg.max_text_per_tile {
        // Tallest first; ties keep arrival order.
        let mut idx: Vec<usize> = (0..buf.text.len()).collect();
        idx.sort_by(|&a, &b| {
            buf.text[b]
                .height
                .partial_cmp(&buf.text[a].height)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(cfg.max_text_per_tile);
        idx.sort_unstable();
        let before = buf.text.len();
        let mut keep = vec![false; before];
        for i in idx {
            keep[i] = true;
        }
        let mut it = keep.iter();
        buf.text.retain(|_| *it.next().unwrap_or(&false));
        ctx.report.dropped_for_text_budget += (before - buf.text.len()) as u64;
    }

    ctx.report.points += buf.lines.iter().map(|l| l.len() as u64).sum::<u64>();
    ctx.report.text += buf.text.len() as u64;

    let mut layers = Vec::new();
    if !buf.lines.is_empty() {
        layers.push(Layer {
            name: "lines".into(),
            extent: cfg.extent,
            features: vec![Feature {
                id: None,
                geometry: Geometry::Lines(buf.lines),
                properties: Vec::new(),
            }],
        });
    }
    if !buf.text.is_empty() {
        layers.push(Layer {
            name: "text".into(),
            extent: cfg.extent,
            features: buf
                .text
                .into_iter()
                .map(|t| Feature {
                    id: None,
                    geometry: Geometry::Points(vec![t.at]),
                    properties: vec![
                        ("text".into(), Value::String(t.text)),
                        ("height".into(), Value::Double(t.height)),
                        ("h0".into(), Value::Double(t.h0)),
                        ("rotation".into(), Value::Double(t.rotation)),
                    ],
                })
                .collect(),
        });
    }
    Tile { layers }
}

/// The layers every tiler archive declares.
#[must_use]
pub fn layer_schema() -> Vec<LayerSchema> {
    vec![
        LayerSchema::new("lines"),
        LayerSchema::new("text")
            .with_field("text", FieldType::String)
            .with_field("height", FieldType::Number)
            .with_field("h0", FieldType::Number)
            .with_field("rotation", FieldType::Number),
    ]
}

/// Tiles `primitives` into a PMTiles archive at `path`.
///
/// The archive is written beside `path` and renamed into place only when every
/// tile is in, so a failed or cancelled run leaves nothing at `path`.
///
/// # Errors
///
/// [`TilerError`].
pub fn build_archive(
    primitives: &[Primitive],
    config: &TilerConfig,
    path: impl AsRef<Path>,
    name: &str,
) -> Result<TilerReport, TilerError> {
    config.validate()?;
    build_archive_guarded(
        primitives,
        config,
        path.as_ref(),
        name,
        &mut Guard::new(config),
    )
}

fn build_archive_guarded(
    primitives: &[Primitive],
    config: &TilerConfig,
    path: &Path,
    name: &str,
    guard: &mut Guard,
) -> Result<TilerReport, TilerError> {
    // The world has to be known before the writer is opened, because the
    // header carries the bounds. A cheap first pass finds them.
    let mut all = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for p in primitives {
        guard.check()?;
        if let Some(b) =
            primitive_bounds(p, config.max_text_bytes).filter(|b| b.iter().all(|v| v.is_finite()))
        {
            all[0] = all[0].min(b[0]);
            all[1] = all[1].min(b[1]);
            all[2] = all[2].max(b[2]);
            all[3] = all[3].max(b[3]);
        }
    }
    if all[0] > all[2] {
        return Err(TilerError::NoGeometry);
    }
    let world = WorldTransform::from_bounds(all, config.margin)?;
    let (lon0, lat0) = world.lon_lat([all[0], all[1]]);
    let (lon1, lat1) = world.lon_lat([all[2], all[3]]);
    let mut options = VectorArchiveOptions::new(layer_schema())
        .with_name(name)
        .with_bounds_degrees([lon0, lat0.min(lat1), lon1, lat0.max(lat1)])
        .with_center_degrees((lon0 + lon1) / 2.0, (lat0 + lat1) / 2.0)
        .with_metadata_entry(
            TILER_METADATA_KEY,
            json!({
                "world_side": world.side(),
                "centre": world.centre(),
                "extent": config.extent,
                "buffer": config.buffer,
                "max_zoom": config.max_zoom,
                "text_h0_basis_px": 512,
            }),
        );
    options = options.with_center_zoom(0);
    let mut writer = VectorArchiveWriter::create(path, &options)?;
    let (report, _) = tile_guarded(primitives, config, &mut writer, guard)?;
    writer.finish()?;
    Ok(report)
}

/// Decodes one view of a drawing and tiles it into an archive.
///
/// # Errors
///
/// [`TilerError`], including the decoder's own.
pub fn build_archive_from_view(
    drawing: &dyn CadDrawing,
    view: u32,
    config: &TilerConfig,
    path: impl AsRef<Path>,
    name: &str,
) -> Result<(TilerReport, DecodeReport), TilerError> {
    config.validate()?;
    let mut guard = Guard::new(config);
    let (held, decode) = decode_view_guarded(drawing, view, config, &mut guard)?;
    let report = build_archive_guarded(&held, config, path.as_ref(), name, &mut guard)?;
    Ok((report, decode))
}

/// Decodes one view and tiles it exactly as [`build_archive_from_view`] would,
/// but keeps nothing: every tile is sized (gzipped, as the archive would store
/// it) and dropped. The reports are the same ones a build returns, so a
/// drawing can be checked for fidelity, oversize tiles and budget trouble
/// without a file being written.
///
/// # Errors
///
/// [`TilerError`], including the decoder's own, the deadline and the cancel.
pub fn report_view(
    drawing: &dyn CadDrawing,
    view: u32,
    config: &TilerConfig,
) -> Result<(TilerReport, DecodeReport), TilerError> {
    config.validate()?;
    let mut guard = Guard::new(config);
    let (held, decode) = decode_view_guarded(drawing, view, config, &mut guard)?;
    let (report, _) = tile_guarded(&held, config, &mut MeasureTiles, &mut guard)?;
    Ok((report, decode))
}

/// A [`TileOutput`] that sizes each tile as the archive would store it and
/// keeps nothing.
#[derive(Debug, Default, Clone, Copy)]
pub struct MeasureTiles;

impl TileOutput for MeasureTiles {
    fn put(&mut self, _z: u8, _x: u32, _y: u32, tile: &Tile) -> Result<Option<usize>, TilerError> {
        if tile.is_empty() {
            return Ok(None);
        }
        let encoded = tile.encode().map_err(VectorError::from)?;
        let stored = crate::vector::archive::gzip(&encoded).map_err(VectorError::from)?;
        Ok(Some(stored.len()))
    }
}

fn decode_view_guarded(
    drawing: &dyn CadDrawing,
    view: u32,
    config: &TilerConfig,
    guard: &mut Guard,
) -> Result<(Vec<Primitive>, DecodeReport), TilerError> {
    let views = drawing.views()?;
    let info = views
        .iter()
        .find(|v| v.index() == view)
        .ok_or(CadError::NoSuchView {
            index: view,
            views: views.len() as u32,
        })?;
    if info.kind() != ViewKind::Model {
        return Err(TilerError::ViewNotEnabled {
            index: view,
            name: info.name().to_owned(),
        });
    }
    let mut sink = BoundedSink {
        held: Vec::new(),
        limit: config.max_primitives,
        exceeded: false,
        guard: guard.clone(),
        stopped: None,
    };
    let mut decode = DecodeReport::new();
    match drawing.decode(view, &mut sink, &mut decode) {
        Ok(()) => {}
        Err(_) if sink.stopped.is_some() => {
            // The decoder cannot be interrupted from here. This is the first
            // primitive after the deadline or the cancel, and the decode is
            // abandoned at that point.
            return Err(sink.stopped.take().unwrap_or(TilerError::Cancelled));
        }
        Err(_) if sink.exceeded => {
            return Err(TilerError::TooManyPrimitives {
                limit: config.max_primitives,
            });
        }
        Err(error) => {
            return Err(TilerError::DecodeFailed {
                error,
                report: Box::new(decode),
            });
        }
    }
    Ok((sink.held, decode))
}

/// Holds primitives up to a limit, then refuses, so a block inserted a million
/// times is an error before it is an out-of-memory.
struct BoundedSink {
    held: Vec<Primitive>,
    limit: usize,
    exceeded: bool,
    guard: Guard,
    stopped: Option<TilerError>,
}

impl PrimitiveSink for BoundedSink {
    fn primitive(&mut self, primitive: Primitive) -> Result<(), CadError> {
        if let Err(e) = self.guard.check() {
            let why = e.to_string();
            self.stopped = Some(e);
            return Err(CadError::sink(why));
        }
        if self.held.len() >= self.limit {
            self.exceeded = true;
            return Err(CadError::sink(format!(
                "more than {} primitives held",
                self.limit
            )));
        }
        self.held.push(primitive);
        Ok(())
    }
}
