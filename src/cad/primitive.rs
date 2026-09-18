//! The eight shapes a decode can produce, and the validating constructors
//! that are the only way to make one.
//!
//! Every type here has private fields. A constructor takes the numbers in the
//! shape a provider already holds them — `[f64; 3]` triples and `&[f64]` runs,
//! which is what the wire carries — checks them once, and hands back a value
//! whose accessors return validated types. Raw arrays in, [`Point3`] and
//! [`Vector3`] out; that is the whole convention, and it is what keeps a
//! provider's translation a field copy rather than a second validation layer.

use crate::cad::CadError;
use core::fmt;

/// The backing file's handle for whatever produced a primitive.
///
/// A newtype rather than a bare `u64` so it cannot be swapped with a count, an
/// index or a length by accident, all of which are `u64` on this boundary too.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ItemHandle(u64);

impl ItemHandle {
    /// Wraps a raw handle.
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// The raw handle.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl fmt::Display for ItemHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Where a primitive came from in the drawing.
///
/// One struct rather than the same two fields repeated on eight types. It is
/// what lets a [`Diagnostic`](crate::cad::Diagnostic) about a primitive name
/// the entity a person can find in their CAD application, which is the
/// difference between a report they can act on and a count.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Origin {
    item_handle: Option<ItemHandle>,
    from_expanded_insert: bool,
}

impl Origin {
    /// An origin the drawing gave no handle for.
    pub const UNKNOWN: Self = Self {
        item_handle: None,
        from_expanded_insert: false,
    };

    /// An origin naming the entity that produced the primitive.
    #[must_use]
    pub const fn from_handle(handle: ItemHandle) -> Self {
        Self {
            item_handle: Some(handle),
            from_expanded_insert: false,
        }
    }

    /// The same origin with a handle attached.
    #[must_use]
    pub const fn with_handle(mut self, handle: ItemHandle) -> Self {
        self.item_handle = Some(handle);
        self
    }

    /// The same origin marked as having come from expanding a nested
    /// insertion, so its coordinates were already transformed into the
    /// containing space.
    #[must_use]
    pub const fn with_expanded_insert(mut self, expanded: bool) -> Self {
        self.from_expanded_insert = expanded;
        self
    }

    /// The entity that produced this primitive, if the file had a handle for
    /// it.
    #[must_use]
    pub const fn item_handle(self) -> Option<ItemHandle> {
        self.item_handle
    }

    /// Whether this primitive came from expanding a nested insertion.
    ///
    /// Worth reporting rather than hiding: a drawing whose primitive count is
    /// a hundred times its entity count is one block referenced everywhere,
    /// and that is the shape that makes a tile job run out of memory.
    #[must_use]
    pub const fn is_expanded_insert(self) -> bool {
        self.from_expanded_insert
    }
}

/// A point in drawing units, all three coordinates finite.
///
/// Three dimensions because that is what a drawing carries, and projecting to
/// a plane needs a choice this module is not in a position to make. The tiler
/// makes it.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Point3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Point3 {
    /// The drawing origin.
    pub const ORIGIN: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// Reads three coordinates as a point.
    ///
    /// # Errors
    ///
    /// [`CadError::NonFinite`] when any of them is not finite. One `NaN`
    /// coordinate poisons every bounding-box union above it, and a comparison
    /// against `NaN` is false in both directions, so nothing downstream would
    /// notice.
    ///
    /// ```
    /// use libviprs::cad::Point3;
    ///
    /// assert_eq!(Point3::new(1.0, 2.0, 3.0).unwrap().into_array(), [1.0, 2.0, 3.0]);
    /// assert!(Point3::new(f64::NAN, 0.0, 0.0).is_err());
    /// ```
    pub fn new(x: f64, y: f64, z: f64) -> Result<Self, CadError> {
        Self::from_array([x, y, z])
    }

    /// Reads a coordinate triple as a point, in the shape the wire carries it.
    ///
    /// # Errors
    ///
    /// As [`Point3::new`].
    pub fn from_array(raw: [f64; 3]) -> Result<Self, CadError> {
        point("coordinate", raw)
    }

    /// The x coordinate.
    #[must_use]
    pub const fn x(self) -> f64 {
        self.x
    }

    /// The y coordinate.
    #[must_use]
    pub const fn y(self) -> f64 {
        self.y
    }

    /// The z coordinate.
    #[must_use]
    pub const fn z(self) -> f64 {
        self.z
    }

    /// The three coordinates, in wire order.
    #[must_use]
    pub const fn into_array(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }
}

/// A direction in drawing space: three finite components, never all zero.
///
/// Used for an entity's normal, which is what decides the plane an arc's
/// angles are measured in, and for an ellipse's major axis, which also carries
/// a magnitude. A zero-length normal names no plane, so an arc with one has no
/// meaning at all rather than a slightly wrong one.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Vector3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vector3 {
    /// The +Z direction, which is the normal of everything drawn in plan.
    pub const Z: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 1.0,
    };

    /// Reads three components as a direction.
    ///
    /// The components are kept as given rather than normalised: a provider
    /// hands over the file's own numbers, and rescaling them here would make
    /// the accessors disagree with the drawing for no gain.
    ///
    /// # Errors
    ///
    /// [`CadError::NonFinite`] when a component is not finite, and
    /// [`CadError::ZeroLengthVector`] when all three are zero.
    ///
    /// ```
    /// use libviprs::cad::Vector3;
    ///
    /// assert_eq!(Vector3::from_array([0.0, 0.0, 1.0]).unwrap(), Vector3::Z);
    /// assert!(Vector3::from_array([0.0, 0.0, 0.0]).is_err());
    /// ```
    pub fn from_array(raw: [f64; 3]) -> Result<Self, CadError> {
        vector("vector", raw)
    }

    /// The x component.
    #[must_use]
    pub const fn x(self) -> f64 {
        self.x
    }

    /// The y component.
    #[must_use]
    pub const fn y(self) -> f64 {
        self.y
    }

    /// The z component.
    #[must_use]
    pub const fn z(self) -> f64 {
        self.z
    }

    /// The three components, in wire order.
    #[must_use]
    pub const fn into_array(self) -> [f64; 3] {
        [self.x, self.y, self.z]
    }

    /// The vector's length, which is never zero.
    #[must_use]
    pub fn length(self) -> f64 {
        self.x.hypot(self.y).hypot(self.z)
    }
}

/// Reads a coordinate triple, naming the field it came from in any refusal.
fn point(field: &'static str, raw: [f64; 3]) -> Result<Point3, CadError> {
    match raw.iter().copied().find(|v| !v.is_finite()) {
        Some(value) => Err(CadError::NonFinite { field, value }),
        None => Ok(Point3 {
            x: raw[0],
            y: raw[1],
            z: raw[2],
        }),
    }
}

/// Reads a direction triple, naming the field it came from in any refusal.
fn vector(field: &'static str, raw: [f64; 3]) -> Result<Vector3, CadError> {
    if let Some(value) = raw.iter().copied().find(|v| !v.is_finite()) {
        return Err(CadError::NonFinite { field, value });
    }
    // `-0.0 == 0.0`, so this catches every signed-zero spelling of the zero
    // vector as well as the plain one.
    if raw.iter().all(|v| *v == 0.0) {
        return Err(CadError::ZeroLengthVector { field });
    }
    Ok(Vector3 {
        x: raw[0],
        y: raw[1],
        z: raw[2],
    })
}

/// Reads one scalar, refusing a non-finite value.
fn scalar(field: &'static str, value: f64) -> Result<f64, CadError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(CadError::NonFinite { field, value })
    }
}

/// Reads one scalar that has to be above zero.
fn positive(field: &'static str, value: f64) -> Result<f64, CadError> {
    let value = scalar(field, value)?;
    if value > 0.0 {
        Ok(value)
    } else {
        Err(CadError::NonPositive { field, value })
    }
}

/// Reads a vertex run and its bulges, shared by [`Polyline`] and [`Polygon`].
fn run(
    field: &'static str,
    vertices: &[[f64; 3]],
    bulges: &[f64],
) -> Result<(Vec<Point3>, Vec<f64>), CadError> {
    if vertices.len() < 2 {
        return Err(CadError::TooFewVertices {
            field,
            got: vertices.len(),
            need: 2,
        });
    }
    if !bulges.is_empty() && bulges.len() != vertices.len() {
        return Err(CadError::CountMismatch {
            field: "bulges",
            got: bulges.len(),
            expected: vertices.len(),
            against: "vertices",
        });
    }
    let points = vertices
        .iter()
        .copied()
        .map(|v| point(field, v))
        .collect::<Result<Vec<_>, _>>()?;
    for bulge in bulges {
        scalar("bulge", *bulge)?;
    }
    Ok((points, bulges.to_vec()))
}

/// Two endpoints.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Line {
    origin: Origin,
    start: Point3,
    end: Point3,
}

impl Line {
    /// A line between two points.
    ///
    /// A zero-length line is accepted. It draws nothing, but drawings contain
    /// them and refusing one would report a defect the file does not have;
    /// discarding a degenerate span is the tiler's call, where the pixel size
    /// is known.
    ///
    /// # Errors
    ///
    /// [`CadError::NonFinite`] for a non-finite coordinate in either endpoint.
    pub fn new(start: [f64; 3], end: [f64; 3]) -> Result<Self, CadError> {
        Ok(Self {
            origin: Origin::UNKNOWN,
            start: point("start", start)?,
            end: point("end", end)?,
        })
    }

    /// The same line, attributed to the entity it came from.
    #[must_use]
    pub const fn with_origin(mut self, origin: Origin) -> Self {
        self.origin = origin;
        self
    }

    /// Where this came from.
    #[must_use]
    pub const fn origin(&self) -> Origin {
        self.origin
    }

    /// The first endpoint.
    #[must_use]
    pub const fn start(&self) -> Point3 {
        self.start
    }

    /// The second endpoint.
    #[must_use]
    pub const fn end(&self) -> Point3 {
        self.end
    }
}

/// A vertex run, open or closed, with a normal and a bulge per span.
#[derive(Debug, Clone, PartialEq)]
pub struct Polyline {
    origin: Origin,
    closed: bool,
    normal: Vector3,
    vertices: Vec<Point3>,
    bulges: Vec<f64>,
}

impl Polyline {
    /// Every field the wire carries.
    ///
    /// `bulges` is either empty, meaning every span is straight, or one entry
    /// per vertex: `bulges[i]` belongs to the span from vertex `i` to vertex
    /// `i + 1`, and on a closed run `bulges[n - 1]` is the closing span's. A
    /// bulge is `tan(θ / 4)` for the arc's included angle, positive
    /// counter-clockwise about the normal, and zero is a straight span. They
    /// cross verbatim: a bulge names an exact arc and the run of segments that
    /// would replace it does not, so subdividing one is the tiler's job.
    ///
    /// # Errors
    ///
    /// [`CadError::TooFewVertices`] below two vertices,
    /// [`CadError::CountMismatch`] for a non-empty bulge run of the wrong
    /// length, [`CadError::NonFinite`] for a non-finite vertex or bulge, and
    /// [`CadError::ZeroLengthVector`] for a zero normal.
    pub fn new(
        vertices: &[[f64; 3]],
        closed: bool,
        normal: [f64; 3],
        bulges: &[f64],
    ) -> Result<Self, CadError> {
        let (vertices, bulges) = run("vertex", vertices, bulges)?;
        Ok(Self {
            origin: Origin::UNKNOWN,
            closed,
            normal: vector("normal", normal)?,
            vertices,
            bulges,
        })
    }

    /// An open run of straight spans in the XY plane, which is the common case
    /// and the one a test wants.
    ///
    /// # Errors
    ///
    /// As [`Polyline::new`].
    pub fn straight(vertices: &[[f64; 3]]) -> Result<Self, CadError> {
        Self::new(vertices, false, Vector3::Z.into_array(), &[])
    }

    /// The same polyline, attributed to the entity it came from.
    #[must_use]
    pub fn with_origin(mut self, origin: Origin) -> Self {
        self.origin = origin;
        self
    }

    /// Where this came from.
    #[must_use]
    pub const fn origin(&self) -> Origin {
        self.origin
    }

    /// Whether the last vertex joins back to the first.
    #[must_use]
    pub const fn is_closed(&self) -> bool {
        self.closed
    }

    /// The entity's normal.
    #[must_use]
    pub const fn normal(&self) -> Vector3 {
        self.normal
    }

    /// The vertices, in order.
    #[must_use]
    pub fn vertices(&self) -> &[Point3] {
        &self.vertices
    }

    /// One bulge per vertex, or empty when every span is straight.
    #[must_use]
    pub fn bulges(&self) -> &[f64] {
        &self.bulges
    }

    /// The bulge of the span leaving vertex `index`, zero when there is none.
    ///
    /// Saves every consumer the "is the bulge run empty" branch, which is the
    /// one an arc gets dropped by when somebody forgets it.
    #[must_use]
    pub fn bulge(&self, index: usize) -> f64 {
        self.bulges.get(index).copied().unwrap_or(0.0)
    }
}

/// A closed boundary: the vertex run of a hatch loop or a solid face.
///
/// A [`Polyline`] with `closed` set carries the same numbers, and this is a
/// distinct type rather than a flag because a polygon is what a fill is
/// computed from and an open run is not. The first vertex is not repeated, so
/// the closing span is the one the last bulge describes.
#[derive(Debug, Clone, PartialEq)]
pub struct Polygon {
    origin: Origin,
    normal: Vector3,
    vertices: Vec<Point3>,
    bulges: Vec<f64>,
}

impl Polygon {
    /// Every field the wire carries. Bulges follow [`Polyline::new`]'s rule.
    ///
    /// Two vertices are enough, and that is not an oversight: two vertices
    /// with bulges on both spans is a lens, which is a real boundary. Two
    /// vertices with no bulges is a line traversed twice, which encloses
    /// nothing — a zero-area ring is still a ring, and dropping it is the
    /// tiler's call once it knows the pixel size.
    ///
    /// # Errors
    ///
    /// As [`Polyline::new`].
    pub fn new(vertices: &[[f64; 3]], normal: [f64; 3], bulges: &[f64]) -> Result<Self, CadError> {
        let (vertices, bulges) = run("vertex", vertices, bulges)?;
        Ok(Self {
            origin: Origin::UNKNOWN,
            normal: vector("normal", normal)?,
            vertices,
            bulges,
        })
    }

    /// A ring of straight spans in the XY plane.
    ///
    /// # Errors
    ///
    /// As [`Polygon::new`].
    pub fn straight(vertices: &[[f64; 3]]) -> Result<Self, CadError> {
        Self::new(vertices, Vector3::Z.into_array(), &[])
    }

    /// The same polygon, attributed to the entity it came from.
    #[must_use]
    pub fn with_origin(mut self, origin: Origin) -> Self {
        self.origin = origin;
        self
    }

    /// Where this came from.
    #[must_use]
    pub const fn origin(&self) -> Origin {
        self.origin
    }

    /// The entity's normal.
    #[must_use]
    pub const fn normal(&self) -> Vector3 {
        self.normal
    }

    /// The vertices, in order, with the first not repeated at the end.
    #[must_use]
    pub fn vertices(&self) -> &[Point3] {
        &self.vertices
    }

    /// One bulge per vertex, or empty when every span is straight.
    #[must_use]
    pub fn bulges(&self) -> &[f64] {
        &self.bulges
    }

    /// The bulge of the span leaving vertex `index`, zero when there is none.
    #[must_use]
    pub fn bulge(&self, index: usize) -> f64 {
        self.bulges.get(index).copied().unwrap_or(0.0)
    }
}

/// Centre, radius, and a start and end angle in the plane the normal defines.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Arc {
    origin: Origin,
    centre: Point3,
    radius: f64,
    start_angle: f64,
    end_angle: f64,
    normal: Vector3,
}

impl Arc {
    /// An arc, kept as an arc.
    ///
    /// Angles are in radians, counter-clockwise about `normal`, and the normal
    /// is what decides where angle zero points — the arbitrary axis algorithm
    /// is the recipe, and it belongs to whoever flattens the arc into a plane.
    ///
    /// # Errors
    ///
    /// [`CadError::NonFinite`] for a non-finite centre or angle,
    /// [`CadError::NonPositive`] for a radius at or below zero,
    /// [`CadError::ZeroLengthVector`] for a zero normal, and
    /// [`CadError::EmptySweep`] when the two angles are equal.
    ///
    /// An arc from `0` to `2π` is a full turn and is accepted; an arc from `θ`
    /// to `θ` sweeps nothing. A sweep too small to be visible is *not*
    /// refused, because "too small" needs a tolerance, and a decoder picking
    /// one is the mistake this module exists to avoid.
    pub fn new(
        centre: [f64; 3],
        radius: f64,
        start_angle: f64,
        end_angle: f64,
        normal: [f64; 3],
    ) -> Result<Self, CadError> {
        let start_angle = scalar("start_angle", start_angle)?;
        let end_angle = scalar("end_angle", end_angle)?;
        if start_angle == end_angle {
            return Err(CadError::EmptySweep {
                field: "arc angle",
                value: start_angle,
            });
        }
        Ok(Self {
            origin: Origin::UNKNOWN,
            centre: point("centre", centre)?,
            radius: positive("radius", radius)?,
            start_angle,
            end_angle,
            normal: vector("normal", normal)?,
        })
    }

    /// The same arc, attributed to the entity it came from.
    #[must_use]
    pub const fn with_origin(mut self, origin: Origin) -> Self {
        self.origin = origin;
        self
    }

    /// Where this came from.
    #[must_use]
    pub const fn origin(&self) -> Origin {
        self.origin
    }

    /// The centre.
    #[must_use]
    pub const fn centre(&self) -> Point3 {
        self.centre
    }

    /// The radius, always above zero.
    #[must_use]
    pub const fn radius(&self) -> f64 {
        self.radius
    }

    /// The start angle in radians.
    #[must_use]
    pub const fn start_angle(&self) -> f64 {
        self.start_angle
    }

    /// The end angle in radians.
    #[must_use]
    pub const fn end_angle(&self) -> f64 {
        self.end_angle
    }

    /// The entity's normal.
    #[must_use]
    pub const fn normal(&self) -> Vector3 {
        self.normal
    }

    /// The counter-clockwise sweep in radians, in `(0, 2π]`.
    ///
    /// A full turn comes back as `2π` rather than `0`, because an arc that
    /// closes on itself draws a circle and an arc that sweeps nothing cannot
    /// be built at all.
    ///
    /// ```
    /// use libviprs::cad::Arc;
    /// use std::f64::consts::{PI, TAU};
    ///
    /// let half = Arc::new([0.0; 3], 1.0, 0.0, PI, [0.0, 0.0, 1.0]).unwrap();
    /// assert!((half.sweep() - PI).abs() < 1e-12);
    ///
    /// // Backwards is read the long way round, which is what CCW means.
    /// let most = Arc::new([0.0; 3], 1.0, PI, 0.0, [0.0, 0.0, 1.0]).unwrap();
    /// assert!((most.sweep() - PI).abs() < 1e-12);
    ///
    /// let full = Arc::new([0.0; 3], 1.0, 0.0, TAU, [0.0, 0.0, 1.0]).unwrap();
    /// assert!((full.sweep() - TAU).abs() < 1e-12);
    /// ```
    #[must_use]
    pub fn sweep(&self) -> f64 {
        let span = (self.end_angle - self.start_angle).rem_euclid(std::f64::consts::TAU);
        if span == 0.0 {
            std::f64::consts::TAU
        } else {
            span
        }
    }
}

/// Centre, radius, and a normal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circle {
    origin: Origin,
    centre: Point3,
    radius: f64,
    normal: Vector3,
}

impl Circle {
    /// A circle, kept as a circle.
    ///
    /// # Errors
    ///
    /// [`CadError::NonFinite`] for a non-finite centre,
    /// [`CadError::NonPositive`] for a radius at or below zero, and
    /// [`CadError::ZeroLengthVector`] for a zero normal.
    pub fn new(centre: [f64; 3], radius: f64, normal: [f64; 3]) -> Result<Self, CadError> {
        Ok(Self {
            origin: Origin::UNKNOWN,
            centre: point("centre", centre)?,
            radius: positive("radius", radius)?,
            normal: vector("normal", normal)?,
        })
    }

    /// The same circle, attributed to the entity it came from.
    #[must_use]
    pub const fn with_origin(mut self, origin: Origin) -> Self {
        self.origin = origin;
        self
    }

    /// Where this came from.
    #[must_use]
    pub const fn origin(&self) -> Origin {
        self.origin
    }

    /// The centre.
    #[must_use]
    pub const fn centre(&self) -> Point3 {
        self.centre
    }

    /// The radius, always above zero.
    #[must_use]
    pub const fn radius(&self) -> f64 {
        self.radius
    }

    /// The entity's normal.
    #[must_use]
    pub const fn normal(&self) -> Vector3 {
        self.normal
    }
}

/// Centre, major axis, minor-to-major ratio and a parameter range.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ellipse {
    origin: Origin,
    centre: Point3,
    major_axis: Vector3,
    ratio: f64,
    start_param: f64,
    end_param: f64,
    normal: Vector3,
}

impl Ellipse {
    /// An ellipse or elliptical arc, kept as one.
    ///
    /// `major_axis` is the vector from the centre to the end of the major
    /// axis, so it carries the orientation and the half-length together, which
    /// is how the file stores it.
    ///
    /// # Errors
    ///
    /// [`CadError::NonFinite`] for a non-finite number,
    /// [`CadError::ZeroLengthVector`] for a zero major axis or normal,
    /// [`CadError::NonPositive`] for a ratio at or below zero, and
    /// [`CadError::EmptySweep`] when the parameter range is empty.
    ///
    /// A ratio above one is *not* refused. The format's convention is
    /// `(0, 1]`, but a value a hair over one is a rounding artefact rather
    /// than a malformation, and the curve it describes is an ellipse either
    /// way — with the axes reading as swapped, which the tiler handles and a
    /// refusal here would not.
    pub fn new(
        centre: [f64; 3],
        major_axis: [f64; 3],
        ratio: f64,
        start_param: f64,
        end_param: f64,
        normal: [f64; 3],
    ) -> Result<Self, CadError> {
        let start_param = scalar("start_param", start_param)?;
        let end_param = scalar("end_param", end_param)?;
        if start_param == end_param {
            return Err(CadError::EmptySweep {
                field: "ellipse parameter",
                value: start_param,
            });
        }
        Ok(Self {
            origin: Origin::UNKNOWN,
            centre: point("centre", centre)?,
            major_axis: vector("major_axis", major_axis)?,
            ratio: positive("ratio", ratio)?,
            start_param,
            end_param,
            normal: vector("normal", normal)?,
        })
    }

    /// The same ellipse, attributed to the entity it came from.
    #[must_use]
    pub const fn with_origin(mut self, origin: Origin) -> Self {
        self.origin = origin;
        self
    }

    /// Where this came from.
    #[must_use]
    pub const fn origin(&self) -> Origin {
        self.origin
    }

    /// The centre.
    #[must_use]
    pub const fn centre(&self) -> Point3 {
        self.centre
    }

    /// The vector from the centre to the end of the major axis.
    #[must_use]
    pub const fn major_axis(&self) -> Vector3 {
        self.major_axis
    }

    /// Minor over major, always above zero.
    #[must_use]
    pub const fn ratio(&self) -> f64 {
        self.ratio
    }

    /// The start of the parameter range.
    #[must_use]
    pub const fn start_param(&self) -> f64 {
        self.start_param
    }

    /// The end of the parameter range.
    #[must_use]
    pub const fn end_param(&self) -> f64 {
        self.end_param
    }

    /// The entity's normal.
    #[must_use]
    pub const fn normal(&self) -> Vector3 {
        self.normal
    }
}

/// Degree, knots, control points and weights, untessellated.
#[derive(Debug, Clone, PartialEq)]
pub struct Spline {
    origin: Origin,
    degree: u32,
    flags: u32,
    knots: Vec<f64>,
    controls: Vec<Point3>,
    weights: Vec<f64>,
}

impl Spline {
    /// Bit 0 of the flag word: the spline closes on itself.
    pub const CLOSED: u32 = 1;
    /// Bit 1: the spline is rational, so its weights are meaningful.
    pub const RATIONAL: u32 = 2;
    /// Bit 2: the spline is periodic.
    pub const PERIODIC: u32 = 4;

    /// A B-spline, kept as a B-spline.
    ///
    /// `flags` crosses verbatim, including bits this build has no name for. It
    /// is not cross-checked against `weights`: a writer may set
    /// [`Spline::RATIONAL`] and omit unit weights, or carry weights on a
    /// non-rational curve, and neither makes the curve unevaluable.
    /// [`Spline::weights`] being non-empty is what a consumer should branch
    /// on.
    ///
    /// # Errors
    ///
    /// [`CadError::SplineDegree`] for a degree of zero,
    /// [`CadError::SplineControlCount`] below `degree + 1` control points,
    /// [`CadError::SplineKnotCount`] below `controls + 1` knots,
    /// [`CadError::KnotsDecrease`] for a knot vector that turns around,
    /// [`CadError::CountMismatch`] for a non-empty weight run of the wrong
    /// length, [`CadError::NonPositive`] for a weight at or below zero, and
    /// [`CadError::NonFinite`] for any non-finite number.
    ///
    /// The knot floor is `controls + 1` rather than the clamped identity
    /// `controls + degree + 1`, because a periodic spline is legitimately
    /// written both ways and refusing one form would drop curves a file
    /// describes correctly. `controls + 1` is the floor no representation goes
    /// below: fewer knots than that leaves no parameter interval at all.
    pub fn new(
        degree: u32,
        flags: u32,
        knots: &[f64],
        controls: &[[f64; 3]],
        weights: &[f64],
    ) -> Result<Self, CadError> {
        if degree == 0 {
            return Err(CadError::SplineDegree { degree });
        }
        let need_controls = degree as usize + 1;
        if controls.len() < need_controls {
            return Err(CadError::SplineControlCount {
                degree,
                need: need_controls,
                got: controls.len(),
            });
        }
        let need_knots = controls.len() + 1;
        if knots.len() < need_knots {
            return Err(CadError::SplineKnotCount {
                got: knots.len(),
                controls: controls.len(),
                need: need_knots,
            });
        }
        if !weights.is_empty() && weights.len() != controls.len() {
            return Err(CadError::CountMismatch {
                field: "weights",
                got: weights.len(),
                expected: controls.len(),
                against: "control points",
            });
        }

        let mut previous = f64::NEG_INFINITY;
        for (index, knot) in knots.iter().copied().enumerate() {
            let knot = scalar("knot", knot)?;
            if knot < previous {
                return Err(CadError::KnotsDecrease {
                    index,
                    previous,
                    found: knot,
                });
            }
            previous = knot;
        }
        for weight in weights {
            positive("spline weight", *weight)?;
        }

        Ok(Self {
            origin: Origin::UNKNOWN,
            degree,
            flags,
            knots: knots.to_vec(),
            controls: controls
                .iter()
                .copied()
                .map(|c| point("control point", c))
                .collect::<Result<Vec<_>, _>>()?,
            weights: weights.to_vec(),
        })
    }

    /// The same spline, attributed to the entity it came from.
    #[must_use]
    pub fn with_origin(mut self, origin: Origin) -> Self {
        self.origin = origin;
        self
    }

    /// Where this came from.
    #[must_use]
    pub const fn origin(&self) -> Origin {
        self.origin
    }

    /// The degree, always at least one.
    #[must_use]
    pub const fn degree(&self) -> u32 {
        self.degree
    }

    /// The flag word as the file carried it.
    #[must_use]
    pub const fn raw_flags(&self) -> u32 {
        self.flags
    }

    /// Whether [`Spline::CLOSED`] is set.
    #[must_use]
    pub const fn is_closed(&self) -> bool {
        self.flags & Self::CLOSED != 0
    }

    /// Whether [`Spline::RATIONAL`] is set.
    #[must_use]
    pub const fn is_rational(&self) -> bool {
        self.flags & Self::RATIONAL != 0
    }

    /// Whether [`Spline::PERIODIC`] is set.
    #[must_use]
    pub const fn is_periodic(&self) -> bool {
        self.flags & Self::PERIODIC != 0
    }

    /// The knot vector, non-decreasing.
    #[must_use]
    pub fn knots(&self) -> &[f64] {
        &self.knots
    }

    /// The control points.
    #[must_use]
    pub fn controls(&self) -> &[Point3] {
        &self.controls
    }

    /// One weight per control point, or empty.
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }
}

/// A position, a height, a rotation and the text a person reads.
///
/// The content is the *decoded, user-visible* string. See the module
/// documentation for the contract, and [`Text::new`] for what it refuses.
#[derive(Debug, Clone, PartialEq)]
pub struct Text {
    origin: Origin,
    position: Point3,
    height: f64,
    rotation: f64,
    content: String,
}

impl Text {
    /// Text at a point, at a height, at a rotation.
    ///
    /// # Errors
    ///
    /// [`CadError::TextEmpty`] for an empty string and
    /// [`CadError::TextEscapeNotDecoded`] for one carrying an undecoded
    /// `\U+XXXX` or `\M+NXXXX` escape: those two are the text contract, and
    /// the module documentation has the drawings that forced it. Then
    /// [`CadError::NonFinite`] for a non-finite position or rotation and
    /// [`CadError::NonPositive`] for a height at or below zero — a
    /// zero-height text draws nothing, and a file that carries one is telling
    /// you something went wrong upstream.
    ///
    /// ```
    /// use libviprs::cad::{CadError, Text};
    ///
    /// // The decoded string, which is the only thing that crosses.
    /// let text = Text::new([0.0, 0.0, 0.0], 2.5, 0.0, "∅45,6").unwrap();
    /// assert_eq!(text.content(), "∅45,6");
    ///
    /// // The transport escape a reader with no MIF layer hands through.
    /// assert!(matches!(
    ///     Text::new([0.0; 3], 2.5, 0.0, r"\U+220545,6"),
    ///     Err(CadError::TextEscapeNotDecoded { .. })
    /// ));
    ///
    /// // And "I could not read it", which is a diagnostic and not a primitive.
    /// assert!(matches!(
    ///     Text::new([0.0; 3], 2.5, 0.0, ""),
    ///     Err(CadError::TextEmpty)
    /// ));
    ///
    /// // A literal backslash is content, so this is text and not an escape.
    /// assert!(Text::new([0.0; 3], 2.5, 0.0, r"C:\\Users").is_ok());
    /// ```
    pub fn new(
        position: [f64; 3],
        height: f64,
        rotation: f64,
        content: impl Into<String>,
    ) -> Result<Self, CadError> {
        let content = content.into();
        if content.is_empty() {
            return Err(CadError::TextEmpty);
        }
        if let Some(escape) = undecoded_escape(&content) {
            return Err(CadError::TextEscapeNotDecoded {
                escape: escape.to_owned(),
            });
        }
        Ok(Self {
            origin: Origin::UNKNOWN,
            position: point("position", position)?,
            height: positive("height", height)?,
            rotation: scalar("rotation", rotation)?,
            content,
        })
    }

    /// The same text, attributed to the entity it came from.
    #[must_use]
    pub fn with_origin(mut self, origin: Origin) -> Self {
        self.origin = origin;
        self
    }

    /// Where this came from.
    #[must_use]
    pub const fn origin(&self) -> Origin {
        self.origin
    }

    /// The insertion point.
    #[must_use]
    pub const fn position(&self) -> Point3 {
        self.position
    }

    /// The height in drawing units, always above zero.
    #[must_use]
    pub const fn height(&self) -> f64 {
        self.height
    }

    /// The rotation in radians.
    #[must_use]
    pub const fn rotation(&self) -> f64 {
        self.rotation
    }

    /// The decoded, user-visible string, never empty and never carrying an
    /// undecoded transport escape.
    #[must_use]
    pub fn content(&self) -> &str {
        &self.content
    }
}

/// The first undecoded transport escape in `text`, if there is one.
///
/// Two shapes, and both are *transport* rather than content: `\U+XXXX` is MIF
/// and `\M+NXXXX` is CIF, and a writer emits one when a character falls
/// outside the file's code page. A reader with no decoding layer for them
/// hands the escape through as though the drawing said `\U+00B0` rather than
/// `°`.
///
/// `\\` is a literal backslash in MTEXT, so a run of two is skipped and what
/// follows is content. Nothing else is interpreted: MTEXT *formatting* codes
/// (`\P`, `{\fArial|b0|i0;…}`) do not change which characters the drawing
/// holds, so they are not this function's business.
///
/// ```
/// use libviprs::cad::undecoded_escape;
///
/// assert_eq!(undecoded_escape(r"94\U+00B0"), Some(r"\U+00B0"));
/// assert_eq!(undecoded_escape(r"\M+5D0B0"), Some(r"\M+5D0B0"));
/// assert_eq!(undecoded_escape("94°"), None);
///
/// // A paragraph break is formatting, not transport.
/// assert_eq!(undecoded_escape(r"line one\Pline two"), None);
///
/// // An escaped backslash is content, so `U+0041` here is just letters.
/// assert_eq!(undecoded_escape(r"\\U+0041"), None);
/// ```
#[must_use]
pub fn undecoded_escape(text: &str) -> Option<&str> {
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'\\' {
            i += 1;
            continue;
        }
        // `\\` is one literal backslash, so the second one opens nothing.
        if bytes.get(i + 1) == Some(&b'\\') {
            i += 2;
            continue;
        }
        if bytes.get(i + 2) == Some(&b'+') {
            let digits = match bytes[i + 1] {
                b'U' => 4,
                b'M' => 5,
                _ => 0,
            };
            // Every byte of an escape is ASCII, so `i + 3 + digits` is a char
            // boundary whenever the digits are there.
            if digits != 0
                && bytes.len() >= i + 3 + digits
                && bytes[i + 3..i + 3 + digits]
                    .iter()
                    .all(u8::is_ascii_hexdigit)
            {
                return Some(&text[i..i + 3 + digits]);
            }
        }
        i += 1;
    }
    None
}

/// Which shape a [`Primitive`] is, without looking at its payload.
///
/// `#[non_exhaustive]`, so the day a ninth shape lands a downstream match does
/// not stop compiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum PrimitiveKind {
    /// Two endpoints.
    Line,
    /// An open or closed vertex run.
    Polyline,
    /// A circular arc.
    Arc,
    /// A full circle.
    Circle,
    /// An ellipse or elliptical arc.
    Ellipse,
    /// A B-spline.
    Spline,
    /// A closed boundary.
    Polygon,
    /// Text.
    Text,
}

impl PrimitiveKind {
    /// Every kind, in the order [`PrimitiveCounts`](crate::cad::PrimitiveCounts)
    /// reports them.
    pub const ALL: [Self; 8] = [
        Self::Line,
        Self::Polyline,
        Self::Arc,
        Self::Circle,
        Self::Ellipse,
        Self::Spline,
        Self::Polygon,
        Self::Text,
    ];

    /// A lowercase name, for a report a person reads.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Line => "line",
            Self::Polyline => "polyline",
            Self::Arc => "arc",
            Self::Circle => "circle",
            Self::Ellipse => "ellipse",
            Self::Spline => "spline",
            Self::Polygon => "polygon",
            Self::Text => "text",
        }
    }

    /// This kind's slot in a counts array.
    pub(crate) const fn index(self) -> usize {
        match self {
            Self::Line => 0,
            Self::Polyline => 1,
            Self::Arc => 2,
            Self::Circle => 3,
            Self::Ellipse => 4,
            Self::Spline => 5,
            Self::Polygon => 6,
            Self::Text => 7,
        }
    }
}

impl fmt::Display for PrimitiveKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// One shape out of a drawing.
///
/// `#[non_exhaustive]`, because a ninth shape is a change to this crate and
/// not to any file format, and it should not break a consumer's match.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Primitive {
    /// Two endpoints.
    Line(Line),
    /// An open or closed vertex run.
    Polyline(Polyline),
    /// A circular arc.
    Arc(Arc),
    /// A full circle.
    Circle(Circle),
    /// An ellipse or elliptical arc.
    Ellipse(Ellipse),
    /// A B-spline.
    Spline(Spline),
    /// A closed boundary.
    Polygon(Polygon),
    /// Text.
    Text(Text),
}

impl Primitive {
    /// Which shape this is.
    #[must_use]
    pub const fn kind(&self) -> PrimitiveKind {
        match self {
            Self::Line(_) => PrimitiveKind::Line,
            Self::Polyline(_) => PrimitiveKind::Polyline,
            Self::Arc(_) => PrimitiveKind::Arc,
            Self::Circle(_) => PrimitiveKind::Circle,
            Self::Ellipse(_) => PrimitiveKind::Ellipse,
            Self::Spline(_) => PrimitiveKind::Spline,
            Self::Polygon(_) => PrimitiveKind::Polygon,
            Self::Text(_) => PrimitiveKind::Text,
        }
    }

    /// Where this primitive came from in the drawing.
    #[must_use]
    pub const fn origin(&self) -> Origin {
        match self {
            Self::Line(p) => p.origin,
            Self::Polyline(p) => p.origin,
            Self::Arc(p) => p.origin,
            Self::Circle(p) => p.origin,
            Self::Ellipse(p) => p.origin,
            Self::Spline(p) => p.origin,
            Self::Polygon(p) => p.origin,
            Self::Text(p) => p.origin,
        }
    }
}

macro_rules! into_primitive {
    ($($ty:ident),+ $(,)?) => {
        $(
            impl From<$ty> for Primitive {
                fn from(value: $ty) -> Self {
                    Self::$ty(value)
                }
            }
        )+
    };
}

into_primitive!(Line, Polyline, Arc, Circle, Ellipse, Spline, Polygon, Text);

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{PI, TAU};

    /// A `NaN` coordinate is refused, and the refusal names the field.
    ///
    /// This is the one that matters most: a `NaN` compares false against
    /// everything, so a bounding box built from one is silently unusable and
    /// no comparison downstream notices.
    #[test]
    fn a_non_finite_coordinate_never_becomes_a_primitive() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = Line::new([0.0, 0.0, 0.0], [bad, 1.0, 2.0])
                .expect_err("a non-finite endpoint is not a line");
            assert!(
                matches!(err, CadError::NonFinite { field: "end", .. }),
                "{bad} came back as {err:?} rather than a NonFinite naming `end`"
            );
        }

        // The control: the same call with a finite value is a line, so the
        // assertions above cannot be passing because `Line::new` refuses
        // everything.
        assert!(Line::new([0.0, 0.0, 0.0], [1.0, 1.0, 2.0]).is_ok());
    }

    /// A zero normal is refused, including the signed-zero spelling of it.
    #[test]
    fn a_zero_length_normal_defines_no_plane_and_is_refused() {
        let err = Circle::new([0.0; 3], 1.0, [0.0, 0.0, 0.0])
            .expect_err("a circle with no normal has no plane");
        assert!(matches!(
            err,
            CadError::ZeroLengthVector { field: "normal" }
        ));

        let signed = Circle::new([0.0; 3], 1.0, [-0.0, 0.0, -0.0])
            .expect_err("negative zero is still zero length");
        assert!(matches!(
            signed,
            CadError::ZeroLengthVector { field: "normal" }
        ));

        assert!(Circle::new([0.0; 3], 1.0, [0.0, 0.0, 1.0]).is_ok());
    }

    /// A radius has to be above zero, and zero is on the wrong side of it.
    #[test]
    fn a_radius_at_or_below_zero_is_refused() {
        for radius in [0.0, -1.0, -f64::MIN_POSITIVE] {
            let err = Circle::new([0.0; 3], radius, [0.0, 0.0, 1.0])
                .expect_err("a non-positive radius is not a circle");
            assert!(
                matches!(
                    err,
                    CadError::NonPositive {
                        field: "radius",
                        ..
                    }
                ),
                "radius {radius} came back as {err:?}"
            );
        }
        assert!(Circle::new([0.0; 3], f64::MIN_POSITIVE, [0.0, 0.0, 1.0]).is_ok());
    }

    /// An arc that starts and ends at the same angle sweeps nothing.
    ///
    /// The negative control is the full turn: `0` to `2π` has the same
    /// *modular* sweep and must be accepted, so a check written as
    /// `sweep % TAU == 0` would fail here.
    #[test]
    fn an_arc_from_an_angle_to_itself_is_refused_but_a_full_turn_is_not() {
        let err = Arc::new([0.0; 3], 1.0, PI, PI, [0.0, 0.0, 1.0])
            .expect_err("an arc of no sweep draws nothing");
        assert!(matches!(err, CadError::EmptySweep { .. }));

        let full =
            Arc::new([0.0; 3], 1.0, 0.0, TAU, [0.0, 0.0, 1.0]).expect("a full turn is a real arc");
        assert!(
            (full.sweep() - TAU).abs() < 1e-12,
            "a full turn's sweep is 2π, not {}",
            full.sweep()
        );
    }

    /// An ellipse with an empty parameter range is refused, and so is a zero
    /// major axis; a ratio over one is not.
    #[test]
    fn an_ellipse_is_refused_for_an_empty_range_and_a_zero_axis() {
        assert!(matches!(
            Ellipse::new([0.0; 3], [1.0, 0.0, 0.0], 0.5, 1.0, 1.0, [0.0, 0.0, 1.0]),
            Err(CadError::EmptySweep { .. })
        ));
        assert!(matches!(
            Ellipse::new([0.0; 3], [0.0; 3], 0.5, 0.0, TAU, [0.0, 0.0, 1.0]),
            Err(CadError::ZeroLengthVector {
                field: "major_axis"
            })
        ));
        assert!(matches!(
            Ellipse::new([0.0; 3], [1.0, 0.0, 0.0], 0.0, 0.0, TAU, [0.0, 0.0, 1.0]),
            Err(CadError::NonPositive { field: "ratio", .. })
        ));

        // Documented as accepted: a ratio a hair over one is a rounding
        // artefact, not a malformation.
        assert!(
            Ellipse::new(
                [0.0; 3],
                [1.0, 0.0, 0.0],
                1.000_000_000_1,
                0.0,
                TAU,
                [0.0, 0.0, 1.0]
            )
            .is_ok(),
            "refusing a ratio marginally above one would drop real curves"
        );
    }

    /// A polyline needs two vertices, and its bulge run has to match.
    #[test]
    fn a_polyline_needs_two_vertices_and_a_matching_bulge_run() {
        assert!(matches!(
            Polyline::straight(&[[0.0, 0.0, 0.0]]),
            Err(CadError::TooFewVertices {
                got: 1,
                need: 2,
                ..
            })
        ));
        assert!(matches!(
            Polyline::straight(&[]),
            Err(CadError::TooFewVertices {
                got: 0,
                need: 2,
                ..
            })
        ));

        let vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]];
        assert!(matches!(
            Polyline::new(&vertices, true, [0.0, 0.0, 1.0], &[0.5, 0.0]),
            Err(CadError::CountMismatch {
                field: "bulges",
                got: 2,
                expected: 3,
                ..
            })
        ));

        let ok = Polyline::new(&vertices, true, [0.0, 0.0, 1.0], &[0.5, 0.0, 0.0])
            .expect("one bulge per vertex is the shape the wire carries");
        assert_eq!(ok.bulge(0), 0.5);
        assert_eq!(ok.bulge(2), 0.0);

        // An empty bulge run means every span is straight, and `bulge` has to
        // answer for it rather than panic or force a branch on every caller.
        let straight = Polyline::straight(&vertices).expect("three points is a polyline");
        assert!(straight.bulges().is_empty());
        assert_eq!(straight.bulge(1), 0.0);
        assert_eq!(straight.bulge(99), 0.0);
    }

    /// The bulges cross verbatim rather than as the arcs they stand for.
    ///
    /// The epic's whole reason for keeping curves as curves: a bulge names an
    /// exact arc, and a run of segments replacing it needs a tolerance nobody
    /// here is entitled to pick.
    #[test]
    fn a_bulge_is_carried_bit_for_bit() {
        let bulge = 0.414_213_562_373_095_1;
        let line = Polyline::new(
            &[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
            false,
            [0.0, 0.0, 1.0],
            &[bulge, 0.0],
        )
        .expect("a bulged two-point polyline is one arc");

        assert_eq!(
            line.bulges()[0].to_bits(),
            bulge.to_bits(),
            "a bulge that is not bit-identical has been through an arithmetic \
             step the decoder had no tolerance to choose"
        );
    }

    /// A spline's counts have to be able to describe a curve.
    #[test]
    fn a_spline_with_counts_that_describe_nothing_is_refused() {
        let controls = [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 0.0, 0.0]];
        let knots = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0];

        assert!(matches!(
            Spline::new(0, 0, &knots, &controls, &[]),
            Err(CadError::SplineDegree { degree: 0 })
        ));
        assert!(matches!(
            Spline::new(5, 0, &knots, &controls, &[]),
            Err(CadError::SplineControlCount {
                degree: 5,
                need: 6,
                got: 3
            })
        ));
        assert!(matches!(
            Spline::new(2, 0, &[0.0, 1.0], &controls, &[]),
            Err(CadError::SplineKnotCount {
                got: 2,
                controls: 3,
                need: 4
            })
        ));
        assert!(matches!(
            Spline::new(2, 0, &[0.0, 0.0, 0.0, 1.0, 0.5, 1.0], &controls, &[]),
            Err(CadError::KnotsDecrease { index: 4, .. })
        ));
        assert!(matches!(
            Spline::new(2, Spline::RATIONAL, &knots, &controls, &[1.0, 1.0]),
            Err(CadError::CountMismatch {
                field: "weights",
                got: 2,
                expected: 3,
                ..
            })
        ));
        assert!(matches!(
            Spline::new(2, Spline::RATIONAL, &knots, &controls, &[1.0, 0.0, 1.0]),
            Err(CadError::NonPositive {
                field: "spline weight",
                ..
            })
        ));

        let ok = Spline::new(2, Spline::RATIONAL, &knots, &controls, &[1.0, 0.5, 1.0])
            .expect("a clamped quadratic with three controls is a spline");
        assert_eq!(ok.degree(), 2);
        assert!(ok.is_rational());
        assert_eq!(ok.knots().len(), 6);
    }

    /// Flags cross verbatim, including bits this build has no name for.
    #[test]
    fn spline_flags_survive_bits_this_build_does_not_name() {
        let controls = [[0.0; 3], [1.0, 1.0, 0.0]];
        let knots = [0.0, 0.0, 1.0, 1.0];
        let flags = Spline::CLOSED | 0x8000_0000;

        let spline = Spline::new(1, flags, &knots, &controls, &[]).expect("a degree 1 spline");
        assert_eq!(
            spline.raw_flags(),
            flags,
            "an unknown flag bit has to survive, or a later wire version's \
             meaning is lost before anybody can read it"
        );
        assert!(spline.is_closed());
        assert!(!spline.is_periodic());
    }

    /// The text contract, in the types. Row 19 of the review that produced
    /// this module: two readers of the same drawing disagreed about what its
    /// text said, and both disagreements were silent.
    #[test]
    fn text_refuses_an_undecoded_transport_escape() {
        // The exact string one reader returned where the other returned
        // "∅45,6".
        let err = Text::new([0.0; 3], 2.5, 0.0, r"\U+220545,6")
            .expect_err("a transport escape is not what the drawing says");
        assert!(
            matches!(&err, CadError::TextEscapeNotDecoded { escape } if escape == r"\U+2205"),
            "{err:?} has to name the escape it found"
        );

        // CIF, the other shape.
        assert!(matches!(
            Text::new([0.0; 3], 2.5, 0.0, r"\M+5D0B0"),
            Err(CadError::TextEscapeNotDecoded { .. })
        ));

        // The decoded string is what crosses.
        let ok = Text::new([0.0; 3], 2.5, 0.0, "∅45,6").expect("decoded text is a primitive");
        assert_eq!(ok.content(), "∅45,6");
    }

    /// Empty is not how a provider says "I could not read this".
    #[test]
    fn text_refuses_the_empty_string() {
        assert!(matches!(
            Text::new([0.0; 3], 2.5, 0.0, ""),
            Err(CadError::TextEmpty)
        ));

        // A single space is content: the drawing says something, even if it is
        // whitespace, and a provider that lost the text has a diagnostic for
        // it rather than a primitive.
        assert!(Text::new([0.0; 3], 2.5, 0.0, " ").is_ok());
    }

    /// Formatting is not transport, and neither is an escaped backslash.
    ///
    /// The negative control for the escape scan. A detector that fired on any
    /// backslash would refuse most real MTEXT, and one that fired on `\\U+`
    /// would refuse a Windows path.
    #[test]
    fn the_escape_scan_leaves_formatting_and_escaped_backslashes_alone() {
        for content in [
            r"line one\Pline two",
            r"{\fArial|b0|i0|c0|p34;plain}",
            r"C:\\Users\\model.dwg",
            r"\U+",
            r"\U+12",
            r"\M+123",
            "94°",
        ] {
            assert_eq!(
                undecoded_escape(content),
                None,
                "{content:?} is content, not an undecoded transport escape"
            );
        }

        // And the positive control, so the loop above cannot be passing
        // against a scan that never fires.
        assert_eq!(undecoded_escape(r"94\U+00B0"), Some(r"\U+00B0"));
        assert_eq!(undecoded_escape(r"\M+5D0B0 trailing"), Some(r"\M+5D0B0"));
    }

    /// A zero-height text draws nothing, so it is a defect and not a shape.
    #[test]
    fn a_text_height_at_or_below_zero_is_refused() {
        assert!(matches!(
            Text::new([0.0; 3], 0.0, 0.0, "A"),
            Err(CadError::NonPositive {
                field: "height",
                ..
            })
        ));
        assert!(matches!(
            Text::new([0.0; 3], -2.0, 0.0, "A"),
            Err(CadError::NonPositive {
                field: "height",
                ..
            })
        ));
    }

    /// Every kind has a slot, and no two kinds share one.
    ///
    /// `index` feeds a fixed-size counts array, so a duplicate would make two
    /// shapes count as one and the report would still look plausible.
    #[test]
    fn every_primitive_kind_has_its_own_counts_slot() {
        let mut seen = [false; PrimitiveKind::ALL.len()];
        for kind in PrimitiveKind::ALL {
            let slot = kind.index();
            assert!(
                !seen[slot],
                "{kind} shares slot {slot} with an earlier kind"
            );
            seen[slot] = true;
        }
        assert!(seen.iter().all(|s| *s), "a slot went unclaimed");
    }

    /// Each shape reports its own kind, and `Primitive::kind` is what a report
    /// counts by.
    #[test]
    fn each_shape_converts_into_a_primitive_of_its_own_kind() {
        let controls = [[0.0; 3], [1.0, 1.0, 0.0]];
        let ring = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]];
        let primitives: Vec<Primitive> = vec![
            Line::new([0.0; 3], [1.0, 0.0, 0.0]).unwrap().into(),
            Polyline::straight(&ring).unwrap().into(),
            Arc::new([0.0; 3], 1.0, 0.0, PI, [0.0, 0.0, 1.0])
                .unwrap()
                .into(),
            Circle::new([0.0; 3], 1.0, [0.0, 0.0, 1.0]).unwrap().into(),
            Ellipse::new([0.0; 3], [2.0, 0.0, 0.0], 0.5, 0.0, TAU, [0.0, 0.0, 1.0])
                .unwrap()
                .into(),
            Spline::new(1, 0, &[0.0, 0.0, 1.0, 1.0], &controls, &[])
                .unwrap()
                .into(),
            Polygon::straight(&ring).unwrap().into(),
            Text::new([0.0; 3], 2.0, 0.0, "A").unwrap().into(),
        ];

        let kinds: Vec<PrimitiveKind> = primitives.iter().map(Primitive::kind).collect();
        assert_eq!(
            kinds,
            PrimitiveKind::ALL.to_vec(),
            "the eight conversions have to land on the eight kinds, in order"
        );
    }

    /// An origin survives the conversion into a `Primitive`, which is the only
    /// way a diagnostic can name the entity a person has to go and look at.
    #[test]
    fn an_origin_reaches_the_primitive_it_was_attached_to() {
        let origin = Origin::from_handle(ItemHandle::new(0x79D)).with_expanded_insert(true);
        let primitive: Primitive = Text::new([0.0; 3], 2.0, 0.0, "my multi line text")
            .unwrap()
            .with_origin(origin)
            .into();

        assert_eq!(
            primitive.origin().item_handle(),
            Some(ItemHandle::new(0x79D))
        );
        assert!(primitive.origin().is_expanded_insert());
        assert_eq!(Origin::UNKNOWN.item_handle(), None);
    }
}
