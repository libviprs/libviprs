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

use serde_json::json;
use thiserror::Error;

use crate::cad::{CadDrawing, CadError, CollectSink, DecodeReport, Primitive};
use crate::vector::archive::{
    FieldType, LayerSchema, VectorArchiveOptions, VectorArchiveWriter, VectorError,
};
use crate::vector::clip::{clip_slab, simplify};
use crate::vector::mvt::{Feature, Geometry, Layer, Point, Tile, Value};
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
    /// What to do with a tile past `max_tile_bytes`.
    pub oversize: OversizePolicy,
}

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
            oversize: OversizePolicy::Report,
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
    #[error("zoom {zoom} needs more than {limit} points held at once")]
    LevelBudgetExceeded {
        /// The zoom.
        zoom: u8,
        /// The configured limit.
        limit: usize,
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
    /// The decoder failed.
    #[error(transparent)]
    Decode(#[from] CadError),
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

fn flatten(p: &Primitive, tol: f64) -> Flat {
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
            text: t.content().to_owned(),
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
fn primitive_bounds(p: &Primitive) -> Option<[f64; 4]> {
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
    match flatten(p, tol) {
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
            // Text has no measured width here: it is estimated from the glyph
            // count, which is enough to keep it inside the world.
            let w = height * 0.6 * text.chars().count() as f64;
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
}

struct Ctx<'a> {
    config: &'a TilerConfig,
    report: &'a mut TilerReport,
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
        if level.points > ctx.config.max_level_points {
            return Err(TilerError::LevelBudgetExceeded {
                zoom: level.zoom,
                limit: ctx.config.max_level_points,
            });
        }
        level.tiles.entry((tx, ty)).or_default().lines.push(pts);
    }
    Ok(())
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
        let b = primitive_bounds(p);
        match b {
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
        };
        let mut ctx = Ctx {
            config,
            report: &mut report,
        };

        for (p, bx) in primitives.iter().zip(&boxes) {
            let Some(bx) = bx else { continue };
            let is_text = matches!(p, Primitive::Text(_));
            if !is_text
                && (bx[2] - bx[0]) < config.cull_units * unit
                && (bx[3] - bx[1]) < config.cull_units * unit
            {
                ctx.report.culled_subpixel += 1;
                continue;
            }
            match flatten(p, tol) {
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
    // The world has to be known before the writer is opened, because the
    // header carries the bounds. A cheap first pass finds them.
    let mut all = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for p in primitives {
        if let Some(b) = primitive_bounds(p).filter(|b| b.iter().all(|v| v.is_finite())) {
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
    let (report, _) = tile_primitives(primitives, config, &mut writer)?;
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
    let mut sink = CollectSink::new();
    let mut decode = DecodeReport::new();
    drawing.decode(view, &mut sink, &mut decode)?;
    let report = build_archive(sink.collected(), config, path, name)?;
    Ok((report, decode))
}
