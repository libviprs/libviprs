//! A Mapbox Vector Tile (spec 2.1) encoder and a bounded decoder.
//!
//! Hand-written protobuf, because a tile is three messages and a handful of
//! varints, and the dependency bar in `CONTRIBUTING.md` is not worth clearing
//! for that. The decoder is here for the same reason the PMTiles reader has a
//! validator: a writer checked only against its own reader round-trips
//! perfectly through a shared misreading, so the decoder exists to be pointed
//! at tiles somebody else produced and, in the tests, at ours.
//!
//! # What a tile is
//!
//! ```text
//! Tile    { Layer* }
//! Layer   { name, extent, Feature* }          keys and values are interned on encode
//! Feature { id?, properties, Geometry }
//! Geometry = Points | Lines | Polygons        integer tile coordinates, y down
//! ```
//!
//! Coordinates are `i32` in tile space: `0..extent` is the tile, and a buffer
//! around it is allowed so a clipped line has somewhere to run to. They are
//! bounded to `|c| < 2^30` ([`COORDINATE_LIMIT`]) so that the difference of two
//! of them, zigzag encoded, always fits the `uint32` the spec's geometry
//! stream is made of.
//!
//! # Polygon winding is the encoder's problem, not the caller's
//!
//! The spec decides exterior from interior by the sign of a ring's area
//! (surveyor's formula, y down): positive is an exterior ring, negative is a
//! hole. [`Polygon`] takes its rings in any orientation and the encoder
//! rewinds them, so the caller cannot produce a tile where a hole is drawn as
//! a second filled island. A ring with no area is refused rather than guessed
//! at.
//!
//! # The decoder is for hostile input
//!
//! Every count it reads is checked against the bytes that remain before
//! anything is reserved, so a few bytes cannot ask for a large allocation, and
//! [`DecodeLimits`] caps the tile, its layers, its features and its points.
//! Nothing panics on any input: the fuzz-style test feeds it every prefix of a
//! real tile and a stream of single-byte mutations.

use std::collections::{HashMap, HashSet};
use std::fmt;

use thiserror::Error;

/// The extent the spec defaults to when a layer does not say.
pub const DEFAULT_EXTENT: u32 = 4096;

/// The largest extent this module will encode or decode.
pub const MAX_EXTENT: u32 = 1 << 24;

/// Coordinates are strictly inside `(-COORDINATE_LIMIT, COORDINATE_LIMIT)`.
///
/// `2^30 - 1` is the largest magnitude whose differences, `2^31 - 2` at most,
/// still zigzag into a `u32`.
pub const COORDINATE_LIMIT: i32 = (1 << 30) - 1;

/// A point in tile coordinates.
pub type Point = [i32; 2];

/// A property value. The seven protobuf value kinds collapse to six here: the
/// spec's `int_value` and `sint_value` are the same thing to a reader.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// A UTF-8 string.
    String(String),
    /// A 32 bit float.
    Float(f32),
    /// A 64 bit float.
    Double(f64),
    /// A signed integer. Encoded as `sint_value`.
    Int(i64),
    /// An unsigned integer. Encoded as `uint_value`.
    Uint(u64),
    /// A boolean.
    Bool(bool),
}

/// A polygon: one exterior ring and any holes, in any orientation.
#[derive(Debug, Clone, PartialEq)]
pub struct Polygon {
    /// The outer ring. The closing point may be repeated or left implicit.
    pub exterior: Vec<Point>,
    /// The holes.
    pub holes: Vec<Vec<Point>>,
}

/// What a feature draws.
#[derive(Debug, Clone, PartialEq)]
pub enum Geometry {
    /// One or more points.
    Points(Vec<Point>),
    /// One or more lines, each with at least two points.
    Lines(Vec<Vec<Point>>),
    /// One or more polygons.
    Polygons(Vec<Polygon>),
}

/// One feature.
#[derive(Debug, Clone, PartialEq)]
pub struct Feature {
    /// The feature id, when it has one.
    pub id: Option<u64>,
    /// What it draws.
    pub geometry: Geometry,
    /// Its properties, in a stable order. Keys must be unique.
    pub properties: Vec<(String, Value)>,
}

/// One layer.
#[derive(Debug, Clone, PartialEq)]
pub struct Layer {
    /// The layer name, which a style addresses it by.
    pub name: String,
    /// The coordinate range `0..extent` the layer's geometry is in.
    pub extent: u32,
    /// The features, in draw order.
    pub features: Vec<Feature>,
}

impl Layer {
    /// An empty layer at the default extent.
    #[must_use]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            extent: DEFAULT_EXTENT,
            features: Vec::new(),
        }
    }
}

/// A whole tile.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Tile {
    /// The layers, in the order they are written.
    pub layers: Vec<Layer>,
}

impl Tile {
    /// Whether no layer holds a feature. Such a tile is valid MVT and not
    /// worth storing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.layers.iter().all(|l| l.features.is_empty())
    }
}

/// Why a tile could not be encoded or decoded.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum MvtError {
    /// A layer has no name, or two layers share one.
    #[error("layer name {name:?} is empty or used twice")]
    LayerName {
        /// The name.
        name: String,
    },
    /// An extent outside `1..=MAX_EXTENT`.
    #[error("layer {layer:?} has extent {extent}, outside 1..={MAX_EXTENT}")]
    Extent {
        /// The layer.
        layer: String,
        /// The extent.
        extent: u32,
    },
    /// A coordinate outside the encodable range.
    #[error("coordinate {value} is outside +-{COORDINATE_LIMIT}")]
    Coordinate {
        /// The value.
        value: i64,
    },
    /// A line with fewer than two points, a ring with fewer than three, or an
    /// empty point set.
    #[error("{what} has {got} points and needs at least {need}")]
    TooFewPoints {
        /// What was too short.
        what: &'static str,
        /// Points found.
        got: usize,
        /// Points needed.
        need: usize,
    },
    /// A ring that encloses no area.
    #[error("a polygon ring encloses no area")]
    DegenerateRing,
    /// A hole in the geometry stream before any exterior ring.
    #[error("a hole ring comes before any exterior ring")]
    HoleWithoutExterior,
    /// A property key that is empty or repeated within a feature.
    #[error("property key {key:?} is empty or repeated in one feature")]
    PropertyKey {
        /// The key.
        key: String,
    },
    /// A float or double that is not finite.
    #[error("property {key:?} is not a finite number")]
    NonFinite {
        /// The key.
        key: String,
    },
    /// A count or length too large for the format.
    #[error("{what} is too large for the format")]
    TooLarge {
        /// What overflowed.
        what: &'static str,
    },
    /// The bytes are not a well-formed tile.
    #[error("malformed tile at byte {at}: {reason}")]
    Malformed {
        /// Offset of the problem, or of the message it was found in.
        at: usize,
        /// What was wrong.
        reason: &'static str,
    },
    /// A decode limit was hit.
    #[error("tile exceeds the decode limit on {what}")]
    Limit {
        /// Which limit.
        what: &'static str,
    },
}

// ---------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------

const MOVE_TO: u32 = 1;
const LINE_TO: u32 = 2;
const CLOSE_PATH: u32 = 7;
/// The command integer keeps its count in 29 bits.
const MAX_COMMAND_COUNT: usize = (1 << 29) - 1;

fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8 & 0x7f) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn put_key(out: &mut Vec<u8>, field: u32, wire: u8) {
    put_varint(out, (u64::from(field) << 3) | u64::from(wire));
}

fn put_bytes(out: &mut Vec<u8>, field: u32, bytes: &[u8]) {
    put_key(out, field, 2);
    put_varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

fn zigzag(v: i64) -> u32 {
    // `v` is a difference of two in-range coordinates, so it is within i32.
    ((v << 1) ^ (v >> 63)) as u32
}

fn unzigzag(v: u32) -> i64 {
    i64::from(v >> 1) ^ -i64::from(v & 1)
}

fn check_coord(p: Point) -> Result<(), MvtError> {
    for c in p {
        if !(-COORDINATE_LIMIT..=COORDINATE_LIMIT).contains(&c) {
            return Err(MvtError::Coordinate {
                value: i64::from(c),
            });
        }
    }
    Ok(())
}

/// Twice the signed area of a ring in surveyor's formula, y down.
fn ring_area2(ring: &[Point]) -> i128 {
    let mut sum = 0i128;
    for i in 0..ring.len() {
        let a = ring[i];
        let b = ring[(i + 1) % ring.len()];
        sum += i128::from(a[0]) * i128::from(b[1]) - i128::from(b[0]) * i128::from(a[1]);
    }
    sum
}

struct Cursor {
    x: i32,
    y: i32,
}

impl Cursor {
    fn push(&mut self, out: &mut Vec<u32>, p: Point) {
        out.push(zigzag(i64::from(p[0]) - i64::from(self.x)));
        out.push(zigzag(i64::from(p[1]) - i64::from(self.y)));
        self.x = p[0];
        self.y = p[1];
    }
}

fn command(id: u32, count: usize) -> Result<u32, MvtError> {
    if count > MAX_COMMAND_COUNT {
        return Err(MvtError::TooLarge {
            what: "a geometry command run",
        });
    }
    Ok(id | ((count as u32) << 3))
}

/// A ring with its closing point removed and its winding set.
fn normalised_ring(ring: &[Point], exterior: bool) -> Result<Vec<Point>, MvtError> {
    let mut pts: Vec<Point> = ring.to_vec();
    if pts.len() > 1 && pts.first() == pts.last() {
        pts.pop();
    }
    if pts.len() < 3 {
        return Err(MvtError::TooFewPoints {
            what: "a polygon ring",
            got: pts.len(),
            need: 3,
        });
    }
    for p in &pts {
        check_coord(*p)?;
    }
    let area = ring_area2(&pts);
    if area == 0 {
        return Err(MvtError::DegenerateRing);
    }
    if (area > 0) != exterior {
        pts.reverse();
    }
    Ok(pts)
}

fn encode_geometry(geometry: &Geometry) -> Result<(u32, Vec<u32>), MvtError> {
    let mut out = Vec::new();
    let mut cur = Cursor { x: 0, y: 0 };
    let kind = match geometry {
        Geometry::Points(points) => {
            if points.is_empty() {
                return Err(MvtError::TooFewPoints {
                    what: "a point set",
                    got: 0,
                    need: 1,
                });
            }
            out.push(command(MOVE_TO, points.len())?);
            for p in points {
                check_coord(*p)?;
                cur.push(&mut out, *p);
            }
            1
        }
        Geometry::Lines(lines) => {
            if lines.is_empty() {
                return Err(MvtError::TooFewPoints {
                    what: "a line set",
                    got: 0,
                    need: 1,
                });
            }
            for line in lines {
                if line.len() < 2 {
                    return Err(MvtError::TooFewPoints {
                        what: "a line",
                        got: line.len(),
                        need: 2,
                    });
                }
                for p in line {
                    check_coord(*p)?;
                }
                out.push(command(MOVE_TO, 1)?);
                cur.push(&mut out, line[0]);
                out.push(command(LINE_TO, line.len() - 1)?);
                for p in &line[1..] {
                    cur.push(&mut out, *p);
                }
            }
            2
        }
        Geometry::Polygons(polygons) => {
            if polygons.is_empty() {
                return Err(MvtError::TooFewPoints {
                    what: "a polygon set",
                    got: 0,
                    need: 1,
                });
            }
            for polygon in polygons {
                let rings = std::iter::once((&polygon.exterior, true))
                    .chain(polygon.holes.iter().map(|h| (h, false)));
                for (ring, exterior) in rings {
                    let pts = normalised_ring(ring, exterior)?;
                    out.push(command(MOVE_TO, 1)?);
                    cur.push(&mut out, pts[0]);
                    out.push(command(LINE_TO, pts.len() - 1)?);
                    for p in &pts[1..] {
                        cur.push(&mut out, *p);
                    }
                    out.push(command(CLOSE_PATH, 1)?);
                }
            }
            3
        }
    };
    Ok((kind, out))
}

fn put_packed_u32(out: &mut Vec<u8>, field: u32, values: &[u32]) {
    let mut body = Vec::with_capacity(values.len() * 2);
    for v in values {
        put_varint(&mut body, u64::from(*v));
    }
    put_bytes(out, field, &body);
}

fn encode_value(value: &Value, key: &str) -> Result<Vec<u8>, MvtError> {
    let mut out = Vec::new();
    match value {
        Value::String(s) => put_bytes(&mut out, 1, s.as_bytes()),
        Value::Float(f) => {
            if !f.is_finite() {
                return Err(MvtError::NonFinite { key: key.into() });
            }
            put_key(&mut out, 2, 5);
            out.extend_from_slice(&f.to_le_bytes());
        }
        Value::Double(d) => {
            if !d.is_finite() {
                return Err(MvtError::NonFinite { key: key.into() });
            }
            put_key(&mut out, 3, 1);
            out.extend_from_slice(&d.to_le_bytes());
        }
        Value::Int(i) => {
            put_key(&mut out, 6, 0);
            put_varint(&mut out, ((*i << 1) ^ (*i >> 63)) as u64);
        }
        Value::Uint(u) => {
            put_key(&mut out, 5, 0);
            put_varint(&mut out, *u);
        }
        Value::Bool(b) => {
            put_key(&mut out, 7, 0);
            put_varint(&mut out, u64::from(*b));
        }
    }
    Ok(out)
}

fn encode_layer(layer: &Layer) -> Result<Vec<u8>, MvtError> {
    if layer.extent == 0 || layer.extent > MAX_EXTENT {
        return Err(MvtError::Extent {
            layer: layer.name.clone(),
            extent: layer.extent,
        });
    }
    let mut keys: Vec<&str> = Vec::new();
    let mut key_ix: HashMap<&str, u32> = HashMap::new();
    let mut values: Vec<Vec<u8>> = Vec::new();
    let mut value_ix: HashMap<Vec<u8>, u32> = HashMap::new();

    let mut features = Vec::new();
    for feature in &layer.features {
        let mut body = Vec::new();
        if let Some(id) = feature.id {
            put_key(&mut body, 1, 0);
            put_varint(&mut body, id);
        }
        let mut seen: HashSet<&str> = HashSet::new();
        let mut tags = Vec::with_capacity(feature.properties.len() * 2);
        for (key, value) in &feature.properties {
            if key.is_empty() || !seen.insert(key.as_str()) {
                return Err(MvtError::PropertyKey { key: key.clone() });
            }
            let k = match key_ix.get(key.as_str()) {
                Some(k) => *k,
                None => {
                    let k = u32::try_from(keys.len()).map_err(|_| MvtError::TooLarge {
                        what: "the key table",
                    })?;
                    keys.push(key);
                    key_ix.insert(key, k);
                    k
                }
            };
            let encoded = encode_value(value, key)?;
            let v = match value_ix.get(&encoded) {
                Some(v) => *v,
                None => {
                    let v = u32::try_from(values.len()).map_err(|_| MvtError::TooLarge {
                        what: "the value table",
                    })?;
                    value_ix.insert(encoded.clone(), v);
                    values.push(encoded);
                    v
                }
            };
            tags.push(k);
            tags.push(v);
        }
        if !tags.is_empty() {
            put_packed_u32(&mut body, 2, &tags);
        }
        let (kind, geometry) = encode_geometry(&feature.geometry)?;
        put_key(&mut body, 3, 0);
        put_varint(&mut body, u64::from(kind));
        put_packed_u32(&mut body, 4, &geometry);
        features.push(body);
    }

    let mut out = Vec::new();
    put_bytes(&mut out, 1, layer.name.as_bytes());
    for f in &features {
        put_bytes(&mut out, 2, f);
    }
    for k in &keys {
        put_bytes(&mut out, 3, k.as_bytes());
    }
    for v in &values {
        put_bytes(&mut out, 4, v);
    }
    put_key(&mut out, 5, 0);
    put_varint(&mut out, u64::from(layer.extent));
    put_key(&mut out, 15, 0);
    put_varint(&mut out, 2);
    Ok(out)
}

impl Tile {
    /// Serialises the tile as MVT version 2, uncompressed.
    ///
    /// The output is deterministic: keys and values are interned in first-use
    /// order and nothing depends on hash iteration order.
    ///
    /// # Errors
    ///
    /// [`MvtError`] for anything the format cannot express: an empty or
    /// repeated layer name, an extent outside range, a coordinate outside
    /// [`COORDINATE_LIMIT`], a line of one point, a ring with no area, a
    /// repeated property key, or a non-finite float.
    pub fn encode(&self) -> Result<Vec<u8>, MvtError> {
        let mut names: HashSet<&str> = HashSet::new();
        let mut out = Vec::new();
        for layer in &self.layers {
            if layer.name.is_empty() || !names.insert(layer.name.as_str()) {
                return Err(MvtError::LayerName {
                    name: layer.name.clone(),
                });
            }
            put_bytes(&mut out, 3, &encode_layer(layer)?);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

/// Ceilings the decoder holds a tile to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeLimits {
    /// The most bytes a tile may be.
    pub max_bytes: usize,
    /// The most layers.
    pub max_layers: usize,
    /// The most features across the tile.
    pub max_features: usize,
    /// The most points across the tile.
    pub max_points: usize,
    /// The most keys or values in one layer's tables.
    pub max_table_entries: usize,
}

impl Default for DecodeLimits {
    fn default() -> Self {
        Self {
            max_bytes: 32 << 20,
            max_layers: 1024,
            max_features: 1 << 22,
            max_points: 1 << 24,
            max_table_entries: 1 << 22,
        }
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
    base: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8], base: usize) -> Self {
        Self { buf, pos: 0, base }
    }

    fn at(&self) -> usize {
        self.base + self.pos
    }

    fn bad<T>(&self, reason: &'static str) -> Result<T, MvtError> {
        Err(MvtError::Malformed {
            at: self.at(),
            reason,
        })
    }

    fn done(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn varint(&mut self) -> Result<u64, MvtError> {
        let mut v = 0u64;
        for shift in (0..70).step_by(7) {
            let Some(&b) = self.buf.get(self.pos) else {
                return self.bad("varint runs off the end");
            };
            self.pos += 1;
            if shift == 63 && b > 1 {
                return self.bad("varint overflows 64 bits");
            }
            v |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        self.bad("varint is longer than ten bytes")
    }

    fn len_delimited(&mut self) -> Result<(&'a [u8], usize), MvtError> {
        let n = self.varint()?;
        let remaining = self.buf.len() - self.pos;
        let n = usize::try_from(n)
            .ok()
            .filter(|n| *n <= remaining)
            .map_or_else(|| self.bad("length runs past the message"), Ok)?;
        let start = self.pos;
        self.pos += n;
        Ok((&self.buf[start..start + n], self.base + start))
    }

    fn skip(&mut self, wire: u8) -> Result<(), MvtError> {
        match wire {
            0 => {
                self.varint()?;
            }
            1 => self.fixed(8)?,
            2 => {
                self.len_delimited()?;
            }
            5 => self.fixed(4)?,
            _ => return self.bad("unsupported wire type"),
        }
        Ok(())
    }

    fn fixed(&mut self, n: usize) -> Result<(), MvtError> {
        if self.buf.len() - self.pos < n {
            return self.bad("fixed-width field runs off the end");
        }
        self.pos += n;
        Ok(())
    }

    fn key(&mut self) -> Result<(u32, u8), MvtError> {
        let k = self.varint()?;
        let field =
            u32::try_from(k >> 3).map_or_else(|_| self.bad("field number too large"), Ok)?;
        Ok((field, (k & 7) as u8))
    }
}

struct Budget {
    features: usize,
    points: usize,
    limits: DecodeLimits,
}

fn decode_value(buf: &[u8], base: usize) -> Result<Value, MvtError> {
    let mut r = Reader::new(buf, base);
    let mut out = None;
    while !r.done() {
        let (field, wire) = r.key()?;
        let value = match (field, wire) {
            (1, 2) => {
                let (b, at) = r.len_delimited()?;
                Value::String(
                    std::str::from_utf8(b)
                        .map_err(|_| MvtError::Malformed {
                            at,
                            reason: "string value is not UTF-8",
                        })?
                        .to_owned(),
                )
            }
            (2, 5) => {
                let start = r.pos;
                r.fixed(4)?;
                let mut a = [0u8; 4];
                a.copy_from_slice(&r.buf[start..start + 4]);
                Value::Float(f32::from_le_bytes(a))
            }
            (3, 1) => {
                let start = r.pos;
                r.fixed(8)?;
                let mut a = [0u8; 8];
                a.copy_from_slice(&r.buf[start..start + 8]);
                Value::Double(f64::from_le_bytes(a))
            }
            (4, 0) => Value::Int(r.varint()? as i64),
            (5, 0) => Value::Uint(r.varint()?),
            (6, 0) => {
                let z = r.varint()?;
                Value::Int(((z >> 1) as i64) ^ -((z & 1) as i64))
            }
            (7, 0) => Value::Bool(r.varint()? != 0),
            _ => {
                r.skip(wire)?;
                continue;
            }
        };
        out = Some(value);
    }
    out.ok_or(MvtError::Malformed {
        at: base,
        reason: "a value message holds no value",
    })
}

fn packed_u32(buf: &[u8], base: usize, into: &mut Vec<u32>) -> Result<(), MvtError> {
    let mut r = Reader::new(buf, base);
    while !r.done() {
        let v = r.varint()?;
        into.push(u32::try_from(v).map_or_else(|_| r.bad("packed value exceeds 32 bits"), Ok)?);
    }
    Ok(())
}

fn decode_geometry(
    kind: u64,
    commands: &[u32],
    at: usize,
    budget: &mut Budget,
) -> Result<Geometry, MvtError> {
    let bad = |reason| MvtError::Malformed { at, reason };
    let mut pos = 0usize;
    let (mut x, mut y) = (0i64, 0i64);
    let mut points: Vec<Point> = Vec::new();
    let mut lines: Vec<Vec<Point>> = Vec::new();
    let mut polygons: Vec<Polygon> = Vec::new();
    let mut ring: Vec<Point> = Vec::new();

    while pos < commands.len() {
        let c = commands[pos];
        pos += 1;
        let (id, count) = (c & 7, (c >> 3) as usize);
        match id {
            MOVE_TO | LINE_TO => {
                let need = count
                    .checked_mul(2)
                    .ok_or_else(|| bad("command count overflows"))?;
                if commands.len() - pos < need {
                    return Err(bad("command asks for more parameters than remain"));
                }
                if count == 0 {
                    return Err(bad("a move or line command with no points"));
                }
                budget.points = budget
                    .points
                    .checked_add(count)
                    .filter(|p| *p <= budget.limits.max_points)
                    .ok_or(MvtError::Limit { what: "points" })?;
                let params = &commands[pos..pos + need];
                pos += need;
                let mut pts = Vec::with_capacity(count);
                for &[px, py] in params.as_chunks::<2>().0 {
                    x += unzigzag(px);
                    y += unzigzag(py);
                    let lim = i64::from(COORDINATE_LIMIT);
                    if !(-lim..=lim).contains(&x) || !(-lim..=lim).contains(&y) {
                        return Err(MvtError::Coordinate {
                            value: if (-lim..=lim).contains(&x) { y } else { x },
                        });
                    }
                    pts.push([x as i32, y as i32]);
                }
                match (kind, id) {
                    (1, MOVE_TO) => points.extend(pts),
                    (1, _) => return Err(bad("a point feature uses LineTo")),
                    (2, MOVE_TO) => {
                        if count != 1 {
                            return Err(bad("a line starts with more than one MoveTo point"));
                        }
                        lines.push(pts);
                    }
                    (2, _) => match lines.last_mut() {
                        Some(line) if !line.is_empty() => line.extend(pts),
                        _ => return Err(bad("LineTo without a fresh MoveTo")),
                    },
                    (3, MOVE_TO) => {
                        if count != 1 || !ring.is_empty() {
                            return Err(bad("a ring starts badly"));
                        }
                        ring = pts;
                    }
                    (3, _) => {
                        if ring.is_empty() {
                            return Err(bad("LineTo without a MoveTo"));
                        }
                        ring.extend(pts);
                    }
                    _ => return Err(bad("unknown geometry type")),
                }
            }
            CLOSE_PATH => {
                if count != 1 || kind != 3 {
                    return Err(bad("ClosePath is only valid once, in a polygon"));
                }
                if ring.len() < 3 {
                    return Err(MvtError::TooFewPoints {
                        what: "a polygon ring",
                        got: ring.len(),
                        need: 3,
                    });
                }
                let area = ring_area2(&ring);
                let done = std::mem::take(&mut ring);
                match area.cmp(&0) {
                    std::cmp::Ordering::Equal => return Err(MvtError::DegenerateRing),
                    std::cmp::Ordering::Greater => polygons.push(Polygon {
                        exterior: done,
                        holes: Vec::new(),
                    }),
                    std::cmp::Ordering::Less => match polygons.last_mut() {
                        Some(p) => p.holes.push(done),
                        None => return Err(MvtError::HoleWithoutExterior),
                    },
                }
            }
            _ => return Err(bad("unknown geometry command")),
        }
    }

    match kind {
        1 if !points.is_empty() => Ok(Geometry::Points(points)),
        2 if !lines.is_empty() && lines.iter().all(|l| l.len() >= 2) => Ok(Geometry::Lines(lines)),
        3 if ring.is_empty() && !polygons.is_empty() => Ok(Geometry::Polygons(polygons)),
        1..=3 => Err(bad("geometry is empty or its last run is unfinished")),
        _ => Err(bad("feature type is not point, line or polygon")),
    }
}

fn decode_feature(
    buf: &[u8],
    base: usize,
    keys: &[String],
    values: &[Value],
    budget: &mut Budget,
) -> Result<Feature, MvtError> {
    let mut r = Reader::new(buf, base);
    let mut id = None;
    let mut tags: Vec<u32> = Vec::new();
    let mut kind = None;
    let mut geometry: Vec<u32> = Vec::new();
    while !r.done() {
        let (field, wire) = r.key()?;
        match (field, wire) {
            (1, 0) => id = Some(r.varint()?),
            (2, 2) => {
                let (b, at) = r.len_delimited()?;
                packed_u32(b, at, &mut tags)?;
            }
            (2, 0) => tags.push(
                u32::try_from(r.varint()?).map_or_else(|_| r.bad("tag exceeds 32 bits"), Ok)?,
            ),
            (3, 0) => kind = Some(r.varint()?),
            (4, 2) => {
                let (b, at) = r.len_delimited()?;
                packed_u32(b, at, &mut geometry)?;
            }
            (4, 0) => geometry.push(
                u32::try_from(r.varint()?)
                    .map_or_else(|_| r.bad("geometry integer exceeds 32 bits"), Ok)?,
            ),
            _ => r.skip(wire)?,
        }
    }
    if !tags.len().is_multiple_of(2) {
        return Err(MvtError::Malformed {
            at: base,
            reason: "tags come in pairs",
        });
    }
    let mut properties = Vec::with_capacity(tags.len() / 2);
    for &[tk, tv] in tags.as_chunks::<2>().0 {
        let (Some(k), Some(v)) = (keys.get(tk as usize), values.get(tv as usize)) else {
            return Err(MvtError::Malformed {
                at: base,
                reason: "a tag points outside the key or value table",
            });
        };
        properties.push((k.clone(), v.clone()));
    }
    let Some(kind) = kind else {
        return Err(MvtError::Malformed {
            at: base,
            reason: "feature has no type",
        });
    };
    Ok(Feature {
        id,
        geometry: decode_geometry(kind, &geometry, base, budget)?,
        properties,
    })
}

fn decode_layer(buf: &[u8], base: usize, budget: &mut Budget) -> Result<Layer, MvtError> {
    let mut r = Reader::new(buf, base);
    let mut name = None;
    let mut extent = DEFAULT_EXTENT;
    let mut version = None;
    let mut features: Vec<(&[u8], usize)> = Vec::new();
    let mut keys: Vec<String> = Vec::new();
    let mut values: Vec<Value> = Vec::new();
    while !r.done() {
        let (field, wire) = r.key()?;
        match (field, wire) {
            (1, 2) => {
                let (b, at) = r.len_delimited()?;
                name = Some(
                    std::str::from_utf8(b)
                        .map_err(|_| MvtError::Malformed {
                            at,
                            reason: "layer name is not UTF-8",
                        })?
                        .to_owned(),
                );
            }
            (2, 2) => {
                budget.features += 1;
                if budget.features > budget.limits.max_features {
                    return Err(MvtError::Limit { what: "features" });
                }
                features.push(r.len_delimited()?);
            }
            (3, 2) => {
                if keys.len() >= budget.limits.max_table_entries {
                    return Err(MvtError::Limit { what: "keys" });
                }
                let (b, at) = r.len_delimited()?;
                keys.push(
                    std::str::from_utf8(b)
                        .map_err(|_| MvtError::Malformed {
                            at,
                            reason: "key is not UTF-8",
                        })?
                        .to_owned(),
                );
            }
            (4, 2) => {
                if values.len() >= budget.limits.max_table_entries {
                    return Err(MvtError::Limit { what: "values" });
                }
                let (b, at) = r.len_delimited()?;
                values.push(decode_value(b, at)?);
            }
            (5, 0) => {
                let e = r.varint()?;
                extent = u32::try_from(e)
                    .ok()
                    .filter(|e| (1..=MAX_EXTENT).contains(e))
                    .map_or_else(|| r.bad("extent out of range"), Ok)?;
            }
            (15, 0) => version = Some(r.varint()?),
            _ => r.skip(wire)?,
        }
    }
    if version != Some(2) {
        return Err(MvtError::Malformed {
            at: base,
            reason: "layer is not version 2",
        });
    }
    let Some(name) = name.filter(|n| !n.is_empty()) else {
        return Err(MvtError::Malformed {
            at: base,
            reason: "layer has no name",
        });
    };
    let mut out = Vec::with_capacity(features.len());
    for (b, at) in features {
        out.push(decode_feature(b, at, &keys, &values, budget)?);
    }
    Ok(Layer {
        name,
        extent,
        features: out,
    })
}

impl Tile {
    /// Parses an uncompressed MVT, version 2.
    ///
    /// Unknown fields are skipped, as the protobuf rules ask. Everything the
    /// spec says a tile must be is checked: layer version and name, tag
    /// indices, geometry command grammar, ring winding.
    ///
    /// # Errors
    ///
    /// [`MvtError`], never a panic, for any input.
    pub fn decode(bytes: &[u8]) -> Result<Self, MvtError> {
        Self::decode_with(bytes, &DecodeLimits::default())
    }

    /// As [`Tile::decode`], under `limits`.
    ///
    /// # Errors
    ///
    /// [`MvtError`], including [`MvtError::Limit`].
    pub fn decode_with(bytes: &[u8], limits: &DecodeLimits) -> Result<Self, MvtError> {
        if bytes.len() > limits.max_bytes {
            return Err(MvtError::Limit { what: "bytes" });
        }
        let mut budget = Budget {
            features: 0,
            points: 0,
            limits: *limits,
        };
        let mut r = Reader::new(bytes, 0);
        let mut layers = Vec::new();
        let mut names = HashSet::new();
        while !r.done() {
            let (field, wire) = r.key()?;
            if (field, wire) == (3, 2) {
                if layers.len() >= limits.max_layers {
                    return Err(MvtError::Limit { what: "layers" });
                }
                let (b, at) = r.len_delimited()?;
                let layer = decode_layer(b, at, &mut budget)?;
                if !names.insert(layer.name.clone()) {
                    return Err(MvtError::LayerName { name: layer.name });
                }
                layers.push(layer);
            } else {
                r.skip(wire)?;
            }
        }
        Ok(Self { layers })
    }
}

impl fmt::Display for Tile {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tile[")?;
        for (i, l) in self.layers.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{}:{}", l.name, l.features.len())?;
        }
        f.write_str("]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(x: i32, y: i32, s: i32) -> Vec<Point> {
        vec![[x, y], [x + s, y], [x + s, y + s], [x, y + s]]
    }

    #[test]
    fn zigzag_round_trips_the_extremes() {
        for v in [
            0i64,
            1,
            -1,
            2,
            -2,
            1 << 30,
            -(1 << 30),
            2_147_483_646,
            -2_147_483_646,
        ] {
            assert_eq!(unzigzag(zigzag(v)), v);
        }
    }

    #[test]
    fn a_polygon_with_a_hole_round_trips_and_is_rewound() {
        // Hole given clockwise and exterior given counter-clockwise on screen,
        // both the wrong way round for the spec.
        let mut ext = square(0, 0, 100);
        ext.reverse();
        let hole = square(20, 20, 10);
        let tile = Tile {
            layers: vec![Layer {
                name: "fills".into(),
                extent: 4096,
                features: vec![Feature {
                    id: Some(7),
                    geometry: Geometry::Polygons(vec![Polygon {
                        exterior: ext,
                        holes: vec![hole],
                    }]),
                    properties: vec![("layer".into(), Value::String("A-WALL".into()))],
                }],
            }],
        };
        let back = Tile::decode(&tile.encode().unwrap()).unwrap();
        let Geometry::Polygons(p) = &back.layers[0].features[0].geometry else {
            panic!("not polygons");
        };
        assert!(ring_area2(&p[0].exterior) > 0, "exterior is positive");
        assert!(ring_area2(&p[0].holes[0]) < 0, "hole is negative");
        assert_eq!(back.layers[0].features[0].id, Some(7));
    }

    #[test]
    fn lines_and_points_round_trip_with_properties() {
        let tile = Tile {
            layers: vec![
                Layer {
                    name: "lines".into(),
                    extent: 8192,
                    features: vec![Feature {
                        id: None,
                        geometry: Geometry::Lines(vec![
                            vec![[0, 0], [10, 10], [-5, 3]],
                            vec![[8000, 8000], [8100, 8100]],
                        ]),
                        properties: vec![],
                    }],
                },
                Layer {
                    name: "text".into(),
                    extent: 8192,
                    features: vec![Feature {
                        id: None,
                        geometry: Geometry::Points(vec![[100, 200]]),
                        properties: vec![
                            ("text".into(), Value::String("A\u{2205}45".into())),
                            ("height".into(), Value::Double(2.5)),
                            ("n".into(), Value::Int(-9)),
                            ("u".into(), Value::Uint(u64::MAX)),
                            ("b".into(), Value::Bool(true)),
                            ("f".into(), Value::Float(1.5)),
                        ],
                    }],
                },
            ],
        };
        assert_eq!(Tile::decode(&tile.encode().unwrap()).unwrap(), tile);
    }

    #[test]
    fn identical_values_are_interned_once() {
        let f = |g| Feature {
            id: None,
            geometry: g,
            properties: vec![("k".into(), Value::String("same".into()))],
        };
        let one = Tile {
            layers: vec![Layer {
                name: "a".into(),
                extent: 4096,
                features: vec![f(Geometry::Points(vec![[1, 1]]))],
            }],
        };
        let mut two = one.clone();
        two.layers[0]
            .features
            .push(f(Geometry::Points(vec![[2, 2]])));
        let (a, b) = (one.encode().unwrap(), two.encode().unwrap());
        let count = |bytes: &[u8]| bytes.windows(4).filter(|w| *w == b"same").count();
        assert_eq!((count(&a), count(&b)), (1, 1));
    }

    #[test]
    fn encoding_is_deterministic() {
        let tile = Tile {
            layers: vec![Layer {
                name: "a".into(),
                extent: 4096,
                features: (0..50)
                    .map(|i| Feature {
                        id: Some(i),
                        geometry: Geometry::Points(vec![[i as i32, 1]]),
                        properties: vec![
                            ("k".into(), Value::Int(i64::from(i as i32 % 3))),
                            ("j".into(), Value::String(format!("v{}", i % 5))),
                        ],
                    })
                    .collect(),
            }],
        };
        assert_eq!(tile.encode().unwrap(), tile.encode().unwrap());
    }

    #[test]
    fn the_encoder_refuses_what_the_format_cannot_say() {
        let one = |g, props| Tile {
            layers: vec![Layer {
                name: "a".into(),
                extent: 4096,
                features: vec![Feature {
                    id: None,
                    geometry: g,
                    properties: props,
                }],
            }],
        };
        let err = |t: Tile| t.encode().unwrap_err();
        assert!(matches!(
            err(one(Geometry::Lines(vec![vec![[0, 0]]]), vec![])),
            MvtError::TooFewPoints { .. }
        ));
        assert_eq!(
            err(one(
                Geometry::Polygons(vec![Polygon {
                    exterior: vec![[0, 0], [5, 5], [10, 10]],
                    holes: vec![]
                }]),
                vec![]
            )),
            MvtError::DegenerateRing
        );
        assert!(matches!(
            err(one(Geometry::Points(vec![[i32::MAX, 0]]), vec![])),
            MvtError::Coordinate { .. }
        ));
        assert!(matches!(
            err(one(
                Geometry::Points(vec![[0, 0]]),
                vec![("x".into(), Value::Double(f64::NAN))]
            )),
            MvtError::NonFinite { .. }
        ));
        assert!(matches!(
            err(one(
                Geometry::Points(vec![[0, 0]]),
                vec![("x".into(), Value::Int(1)), ("x".into(), Value::Int(2))]
            )),
            MvtError::PropertyKey { .. }
        ));
        let mut dup = one(Geometry::Points(vec![[0, 0]]), vec![]);
        dup.layers.push(dup.layers[0].clone());
        assert!(matches!(err(dup), MvtError::LayerName { .. }));
        let mut ext = one(Geometry::Points(vec![[0, 0]]), vec![]);
        ext.layers[0].extent = 0;
        assert!(matches!(err(ext), MvtError::Extent { .. }));
    }

    #[test]
    fn extreme_coordinates_survive() {
        let m = COORDINATE_LIMIT;
        let tile = Tile {
            layers: vec![Layer {
                name: "a".into(),
                extent: 4096,
                features: vec![Feature {
                    id: None,
                    geometry: Geometry::Lines(vec![vec![[-m, -m], [m, m], [-m, m]]]),
                    properties: vec![],
                }],
            }],
        };
        assert_eq!(Tile::decode(&tile.encode().unwrap()).unwrap(), tile);
    }

    #[test]
    fn every_prefix_and_every_single_byte_mutation_decodes_or_errs() {
        let tile = Tile {
            layers: vec![Layer {
                name: "fills".into(),
                extent: 4096,
                features: vec![Feature {
                    id: Some(1),
                    geometry: Geometry::Polygons(vec![Polygon {
                        exterior: square(0, 0, 50),
                        holes: vec![square(10, 10, 5)],
                    }]),
                    properties: vec![("k".into(), Value::String("v".into()))],
                }],
            }],
        };
        let bytes = tile.encode().unwrap();
        for n in 0..bytes.len() {
            let _ = Tile::decode(&bytes[..n]);
        }
        for i in 0..bytes.len() {
            for delta in [1u8, 0x7f, 0x80, 0xff] {
                let mut m = bytes.clone();
                m[i] = m[i].wrapping_add(delta);
                let _ = Tile::decode(&m);
            }
        }
    }

    #[test]
    fn a_few_bytes_cannot_ask_for_a_large_allocation() {
        // A feature whose MoveTo claims 2^29 - 1 points with no parameters.
        let mut feature = Vec::new();
        put_key(&mut feature, 3, 0);
        put_varint(&mut feature, 1);
        put_packed_u32(&mut feature, 4, &[MOVE_TO | (((1u32 << 29) - 1) << 3)]);
        let mut layer = Vec::new();
        put_bytes(&mut layer, 1, b"a");
        put_bytes(&mut layer, 2, &feature);
        put_key(&mut layer, 15, 0);
        put_varint(&mut layer, 2);
        let mut tile = Vec::new();
        put_bytes(&mut tile, 3, &layer);
        assert!(matches!(
            Tile::decode(&tile),
            Err(MvtError::Malformed { .. })
        ));
    }

    #[test]
    fn the_limits_are_enforced() {
        let tile = Tile {
            layers: vec![Layer {
                name: "a".into(),
                extent: 4096,
                features: (0..10)
                    .map(|i| Feature {
                        id: None,
                        geometry: Geometry::Points(vec![[i, i]]),
                        properties: vec![],
                    })
                    .collect(),
            }],
        };
        let bytes = tile.encode().unwrap();
        let tight = |f: fn(&mut DecodeLimits)| {
            let mut l = DecodeLimits::default();
            f(&mut l);
            Tile::decode_with(&bytes, &l).unwrap_err()
        };
        assert!(matches!(
            tight(|l| l.max_features = 3),
            MvtError::Limit { .. }
        ));
        assert!(matches!(
            tight(|l| l.max_points = 3),
            MvtError::Limit { .. }
        ));
        assert!(matches!(tight(|l| l.max_bytes = 3), MvtError::Limit { .. }));
        assert!(matches!(
            tight(|l| l.max_layers = 0),
            MvtError::Limit { .. }
        ));
    }

    #[test]
    fn an_empty_tile_encodes_to_nothing_and_says_so() {
        assert!(Tile::default().is_empty());
        assert_eq!(Tile::default().encode().unwrap(), Vec::<u8>::new());
    }
}
