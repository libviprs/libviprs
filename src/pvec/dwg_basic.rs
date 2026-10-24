//! Exact extraction of the basic DWG entities (N4.2, #1073).
//!
//! A `LINE`, a `LWPOLYLINE`, a 2D `POLYLINE`, a `CIRCLE`, an `ARC` and an `ELLIPSE` each have a
//! closed-form PVec equivalent, so nothing here samples, fits or tessellates. Every construction
//! is a call into `pvec_geometry`, and every tolerance is PVec's numeric policy speaking through
//! the refusal it returns. A refusal is classified (see [`BasicError::classification`]) and gets a
//! one-line report row ([`BasicError::warning`]), never repaired.

use std::fmt;

use pvec_geometry::{
    Circle, Composite, Curve, Ellipse, GeometryError, Line, Placement2, Point2, Polyline, Trimmed,
    Vector2,
};
use pvec_numeric::{MAX_POLYLINE_VERTICES, NumericError};

use super::fidelity::{Basis, Classification, ReasonCode};
use super::provider::{Point3, RawEntity, RawGeometry, RawVertex};

/// Which basic source class an extraction came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BasicKind {
    /// `LINE`.
    Line,
    /// `LWPOLYLINE` or a 2D `POLYLINE`, with or without bulges.
    Polyline,
    /// `CIRCLE`.
    Circle,
    /// `ARC`.
    Arc,
    /// `ELLIPSE` over its whole period.
    Ellipse,
    /// `ELLIPSE` cut to an arc.
    EllipticalArc,
}

/// What one segment of a polyline became.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SegmentKind {
    /// A straight segment (bulge exactly zero).
    Straight,
    /// A counter-clockwise arc (positive bulge).
    ArcCounterClockwise,
    /// A clockwise arc (negative bulge).
    ArcClockwise,
}

/// One basic entity, as PVec carries it plus the source numbers PVec's canonical form moves.
#[derive(Debug, Clone)]
pub struct ExactBasic {
    /// The source class.
    pub kind: BasicKind,
    /// The PVec curve.
    pub curve: Curve,
    /// One entry per polyline segment, the closing segment of a closed polyline last. Empty for
    /// every other kind.
    pub segments: Vec<SegmentKind>,
    /// The drawing's closed flag (polylines), `false` otherwise.
    pub closed: bool,
    /// The angles of an arc, or the parameters of an elliptical arc, exactly as stored, in the
    /// drawing's counter-clockwise convention. PVec's trim is a normalised increasing interval
    /// and the numbers here are what the drawing said.
    pub source_range: Option<(f64, f64)>,
    /// The z of the entity's plane.
    pub elevation: f64,
    /// The outcome under N3.5's classes.
    pub classification: Classification,
}

/// Why a basic entity was refused. Nothing is repaired.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum BasicError {
    /// The entity is not a basic one (or not the class the function extracts).
    NotBasic,
    /// A radius that is zero, negative, not finite or below PVec's floor.
    DegenerateRadius {
        /// The radius as stored.
        radius: f64,
    },
    /// Two consecutive end points PVec treats as one point.
    CoincidentVertices {
        /// The index of the vertex that starts the segment.
        index: usize,
    },
    /// An ellipse ratio PVec refuses.
    DegenerateAxisRatio {
        /// The ratio as stored.
        ratio: f64,
    },
    /// A bulge that is not finite or that PVec refuses.
    DegenerateBulge {
        /// The index of the vertex that carries it.
        index: usize,
        /// The bulge as stored.
        bulge: f64,
    },
    /// PVec refused the rest.
    Degenerate {
        /// Which input.
        field: &'static str,
        /// PVec's refusal, unchanged.
        error: GeometryError,
    },
    /// More polyline vertices than PVec supports.
    TooManyVertices {
        /// The limit.
        limit: usize,
        /// Observed.
        observed: usize,
    },
    /// Different z values on one entity.
    NonPlanar,
    /// An extrusion normal that is not +z.
    ExtrusionNotApplied {
        /// The normal as stored.
        normal: Point3,
    },
}

impl fmt::Display for BasicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BasicError::NotBasic => write!(f, "not a basic entity"),
            BasicError::DegenerateRadius { radius } => {
                write!(f, "radius {radius:?} is not usable")
            }
            BasicError::CoincidentVertices { index } => {
                write!(f, "vertices {index} and {} coincide", index + 1)
            }
            BasicError::DegenerateAxisRatio { ratio } => {
                write!(f, "minor/major ratio {ratio:?} is not usable")
            }
            BasicError::DegenerateBulge { index, bulge } => {
                write!(f, "the bulge {bulge:?} at vertex {index} is not usable")
            }
            BasicError::Degenerate { field, error } => {
                write!(f, "PVec refused `{field}`: {error}")
            }
            BasicError::TooManyVertices { limit, observed } => {
                write!(f, "polyline has {observed} vertices, the limit is {limit}")
            }
            BasicError::NonPlanar => write!(f, "the entity does not lie in one plane"),
            BasicError::ExtrusionNotApplied { normal } => {
                write!(f, "extrusion normal {normal:?} is not +z")
            }
        }
    }
}

impl std::error::Error for BasicError {}

impl BasicError {
    /// What became of the entity under N3.5's six classes, with a stable reason code. Total: a new
    /// variant cannot be added without choosing a code.
    #[must_use]
    pub fn classification(&self, source_class: impl Into<String>) -> Classification {
        let reason = match self {
            BasicError::NotBasic => ReasonCode::UnmappedSourceClass,
            BasicError::DegenerateRadius { .. } => ReasonCode::DegenerateRadius,
            BasicError::CoincidentVertices { .. } => ReasonCode::CoincidentVertices,
            BasicError::DegenerateAxisRatio { .. } => ReasonCode::DegenerateAxisRatio,
            BasicError::DegenerateBulge { .. } => ReasonCode::DegenerateBulge,
            BasicError::Degenerate { .. } => ReasonCode::DegenerateGeometry,
            BasicError::TooManyVertices { .. } => ReasonCode::PolylineLimitExceeded,
            BasicError::NonPlanar => ReasonCode::Geometry3d,
            BasicError::ExtrusionNotApplied { .. } => ReasonCode::ExtrusionNotApplied,
        };
        Classification::refused(reason, source_class)
    }

    /// The row the conversion report (N6.7) carries for this refusal: source class, class tag,
    /// reason code and the detail, on one line.
    #[must_use]
    pub fn warning(&self, source_class: &str) -> String {
        let c = self.classification(source_class);
        let code = c.reason().map_or("", ReasonCode::code);
        format!("{source_class}: {} {code}: {self}", c.class().tag())
    }
}

fn degenerate(field: &'static str, error: GeometryError) -> BasicError {
    BasicError::Degenerate { field, error }
}

fn not_finite(field: &'static str) -> BasicError {
    degenerate(field, GeometryError::Numeric(NumericError::NotFinite))
}

/// A PVec refusal that says two points are one: the length floor, a zero direction, a collapsed
/// trim. Anything else stays a plain refusal.
fn coincident_or(field: &'static str, index: usize, error: GeometryError) -> BasicError {
    match error.as_numeric() {
        Some(
            NumericError::DegenerateLength
            | NumericError::ZeroDirection
            | NumericError::CollapsedTrim,
        ) => BasicError::CoincidentVertices { index },
        _ => degenerate(field, error),
    }
}

fn point(field: &'static str, x: f64, y: f64) -> Result<Point2, BasicError> {
    Point2::checked(x, y).map_err(|error| degenerate(field, error))
}

fn z_of(field: &'static str, z: f64) -> Result<f64, BasicError> {
    if z.is_finite() {
        Ok(z)
    } else {
        Err(not_finite(field))
    }
}

fn check_normal(normal: Point3) -> Result<(), BasicError> {
    if normal == [0.0, 0.0, 1.0] {
        Ok(())
    } else {
        Err(BasicError::ExtrusionNotApplied { normal })
    }
}

fn class_of(entity: &RawEntity) -> &str {
    entity.provenance.class_name.as_str()
}

/// Extracts a `LINE`: a trimmed line from the first end point to the second, the direction and the
/// length PVec's own (`Line::new`, `Point2::distance`). The start point is kept bit for bit.
///
/// # Errors
/// See [`BasicError`].
pub fn extract_line(entity: &RawEntity) -> Result<ExactBasic, BasicError> {
    let RawGeometry::Line { start, end } = &entity.geometry else {
        return Err(BasicError::NotBasic);
    };
    let elevation = z_of("start", start[2])?;
    if z_of("end", end[2])? != elevation {
        return Err(BasicError::NonPlanar);
    }
    let p0 = point("start", start[0], start[1])?;
    let p1 = point("end", end[0], end[1])?;
    let line = Line::new(p0, p0.vector_to(p1)).map_err(|e| coincident_or("line", 0, e))?;
    let trimmed = Trimmed::new(Curve::Line(line), 0.0, p0.distance(p1), true)
        .map_err(|e| coincident_or("line", 0, e))?;
    Ok(ExactBasic {
        kind: BasicKind::Line,
        curve: Curve::Trimmed(trimmed),
        segments: Vec::new(),
        closed: false,
        source_range: None,
        elevation,
        classification: Classification::supported(Basis::Equivalent, class_of(entity)),
    })
}

/// The polyline PVec holds for a run of points, or the pair that is degenerate. `first` is the
/// index of the run's first point among the entity's vertices, so the error names a vertex.
fn polyline_curve(points: Vec<Point2>, first: usize) -> Result<Curve, BasicError> {
    match Polyline::new(points.clone()) {
        Ok(p) => Ok(Curve::Polyline(p)),
        Err(error) => {
            // PVec's own rule decides which pair it is, pair by pair.
            for (k, w) in points.windows(2).enumerate() {
                if Polyline::new(w.to_vec()).is_err() {
                    return Err(coincident_or("vertices", first + k, error));
                }
            }
            Err(degenerate("vertices", error))
        }
    }
}

fn segment_kind(v: &RawVertex) -> SegmentKind {
    if v.bulge == 0.0 {
        SegmentKind::Straight
    } else if v.bulge > 0.0 {
        SegmentKind::ArcCounterClockwise
    } else {
        SegmentKind::ArcClockwise
    }
}

/// Extracts a `LWPOLYLINE` or a 2D `POLYLINE`.
///
/// With every bulge zero the result is PVec's polyline, a closed one carrying a copy of its first
/// vertex as the last. With any bulge it is a composite of polylines (straight runs) and trimmed
/// circles (one per bulged segment, `Trimmed::from_bulge`), declared closed when the drawing's
/// flag is set. The closing segment takes the last vertex's bulge. The last vertex's bulge of an
/// open polyline starts no segment and is ignored, but it must still be a number.
///
/// # Errors
/// See [`BasicError`].
pub fn extract_polyline(entity: &RawEntity) -> Result<ExactBasic, BasicError> {
    let RawGeometry::Polyline {
        vertices,
        closed,
        elevation,
        normal,
    } = &entity.geometry
    else {
        return Err(BasicError::NotBasic);
    };
    check_normal(*normal)?;
    let n = vertices.len();
    if n > MAX_POLYLINE_VERTICES {
        return Err(BasicError::TooManyVertices {
            limit: MAX_POLYLINE_VERTICES,
            observed: n,
        });
    }
    let elevation = z_of("elevation", *elevation)?;
    if n < 2 {
        return Err(degenerate(
            "vertices",
            GeometryError::Numeric(NumericError::Limit),
        ));
    }
    if let Some(index) = vertices.iter().position(|v| !v.bulge.is_finite()) {
        return Err(BasicError::DegenerateBulge {
            index,
            bulge: vertices[index].bulge,
        });
    }
    let pts = vertices
        .iter()
        .map(|v| point("vertex", v.x, v.y))
        .collect::<Result<Vec<_>, _>>()?;
    let count = if *closed { n } else { n - 1 };
    let kinds: Vec<SegmentKind> = vertices[..count].iter().map(segment_kind).collect();

    let curve = if kinds.iter().all(|k| *k == SegmentKind::Straight) {
        let mut run = pts;
        if *closed {
            run.push(run[0]);
        }
        polyline_curve(run, 0)?
    } else {
        let mut parts: Vec<Curve> = Vec::new();
        let mut run: Vec<Point2> = Vec::new();
        let mut run_first = 0;
        for (i, kind) in kinds.iter().enumerate() {
            let (a, b) = (pts[i], pts[(i + 1) % n]);
            if *kind == SegmentKind::Straight {
                if run.is_empty() {
                    run_first = i;
                    run.push(a);
                }
                run.push(b);
            } else {
                if !run.is_empty() {
                    parts.push(polyline_curve(std::mem::take(&mut run), run_first)?);
                }
                let bulge = vertices[i].bulge;
                let arc =
                    Trimmed::from_bulge(a, b, bulge).map_err(|error| match error.as_numeric() {
                        Some(NumericError::DegenerateLength) => {
                            BasicError::CoincidentVertices { index: i }
                        }
                        _ => BasicError::DegenerateBulge { index: i, bulge },
                    })?;
                parts.push(Curve::Trimmed(arc));
            }
        }
        if !run.is_empty() {
            parts.push(polyline_curve(run, run_first)?);
        }
        let mut composite = Composite::new(parts).map_err(|error| degenerate("segments", error))?;
        if *closed {
            composite = composite
                .declare_closed()
                .map_err(|error| degenerate("closed", error))?;
        }
        Curve::Composite(composite)
    };

    let basis = if kinds.iter().all(|k| *k == SegmentKind::Straight) {
        Basis::Structural
    } else {
        Basis::Equivalent
    };
    Ok(ExactBasic {
        kind: BasicKind::Polyline,
        curve,
        segments: kinds,
        closed: *closed,
        source_range: None,
        elevation,
        classification: Classification::supported(basis, class_of(entity)),
    })
}

/// A circle at `centre` with `radius`, framed on the x axis (a DWG circle has no reference
/// direction). A refusal is the radius's.
fn circle_of(centre: &Point3, radius: f64) -> Result<(Circle, f64), BasicError> {
    let elevation = z_of("centre", centre[2])?;
    let origin = point("centre", centre[0], centre[1])?;
    let frame = Placement2::new(origin, Vector2::new(1.0, 0.0))
        .map_err(|error| degenerate("centre", error))?;
    let circle = Circle::new(frame, radius).map_err(|_| BasicError::DegenerateRadius { radius })?;
    Ok((circle, elevation))
}

/// Extracts a `CIRCLE`: centre and radius as stored.
///
/// # Errors
/// See [`BasicError`].
pub fn extract_circle(entity: &RawEntity) -> Result<ExactBasic, BasicError> {
    let RawGeometry::Circle {
        center,
        radius,
        normal,
    } = &entity.geometry
    else {
        return Err(BasicError::NotBasic);
    };
    check_normal(*normal)?;
    let (circle, elevation) = circle_of(center, *radius)?;
    Ok(ExactBasic {
        kind: BasicKind::Circle,
        curve: Curve::Circle(circle),
        segments: Vec::new(),
        closed: false,
        source_range: None,
        elevation,
        classification: Classification::supported(Basis::Structural, class_of(entity)),
    })
}

/// Extracts an `ARC`: a trimmed circle running counter-clockwise from the start angle to the end
/// angle, both radians as stored. An end angle below the start wraps through zero, as the drawing
/// means it, and the stored angles come back unchanged in [`ExactBasic::source_range`].
///
/// # Errors
/// See [`BasicError`].
pub fn extract_arc(entity: &RawEntity) -> Result<ExactBasic, BasicError> {
    let RawGeometry::Arc {
        center,
        radius,
        start_angle,
        end_angle,
        normal,
    } = &entity.geometry
    else {
        return Err(BasicError::NotBasic);
    };
    check_normal(*normal)?;
    let (circle, elevation) = circle_of(center, *radius)?;
    let curve = Curve::trimmed_conic(Curve::Circle(circle), *start_angle, *end_angle, true)
        .map_err(|error| degenerate("angles", error))?;
    Ok(ExactBasic {
        kind: BasicKind::Arc,
        curve,
        segments: Vec::new(),
        closed: false,
        source_range: Some((*start_angle, *end_angle)),
        elevation,
        classification: Classification::supported(Basis::Equivalent, class_of(entity)),
    })
}

/// Extracts an `ELLIPSE`: the major-axis vector and the minor/major ratio go to PVec as stored
/// (`Ellipse::from_major_axis`), then the parameter range as a trim. A range that is the whole
/// period comes back as the untrimmed ellipse, which is PVec's call (`Curve::trimmed_conic`).
///
/// # Errors
/// See [`BasicError`].
pub fn extract_ellipse(entity: &RawEntity) -> Result<ExactBasic, BasicError> {
    let RawGeometry::Ellipse {
        center,
        major_axis,
        minor_ratio,
        start_parameter,
        end_parameter,
        normal,
    } = &entity.geometry
    else {
        return Err(BasicError::NotBasic);
    };
    check_normal(*normal)?;
    let elevation = z_of("centre", center[2])?;
    if z_of("major_axis", major_axis[2])? != 0.0 {
        return Err(BasicError::NonPlanar);
    }
    let origin = point("centre", center[0], center[1])?;
    let major = Vector2::new(major_axis[0], major_axis[1]);
    if !(major.x.is_finite() && major.y.is_finite()) {
        return Err(not_finite("major_axis"));
    }
    let ellipse = match Ellipse::from_major_axis(origin, major, *minor_ratio) {
        Ok(e) => e,
        Err(error) => {
            // PVec reports a bad ratio and a bad axis the same way for a radius, so ask it again
            // with a ratio of one: if that builds, the ratio was the problem.
            return Err(match error.as_numeric() {
                Some(NumericError::ZeroDirection) => degenerate("major_axis", error),
                _ if Ellipse::from_major_axis(origin, major, 1.0).is_ok() => {
                    BasicError::DegenerateAxisRatio {
                        ratio: *minor_ratio,
                    }
                }
                _ => BasicError::DegenerateRadius {
                    radius: major.length(),
                },
            });
        }
    };
    let curve = Curve::trimmed_conic(
        Curve::Ellipse(ellipse),
        *start_parameter,
        *end_parameter,
        true,
    )
    .map_err(|error| degenerate("parameters", error))?;
    let (kind, basis) = if matches!(curve, Curve::Ellipse(_)) {
        (BasicKind::Ellipse, Basis::Structural)
    } else {
        (BasicKind::EllipticalArc, Basis::Equivalent)
    };
    Ok(ExactBasic {
        kind,
        curve,
        segments: Vec::new(),
        closed: false,
        source_range: Some((*start_parameter, *end_parameter)),
        elevation,
        classification: Classification::supported(basis, class_of(entity)),
    })
}

/// Extracts whichever basic class the entity's geometry is.
///
/// # Errors
/// See [`BasicError`]. Anything that is not a line, polyline, circle, arc or ellipse is
/// [`BasicError::NotBasic`].
pub fn extract(entity: &RawEntity) -> Result<ExactBasic, BasicError> {
    match &entity.geometry {
        RawGeometry::Line { .. } => extract_line(entity),
        RawGeometry::Polyline { .. } => extract_polyline(entity),
        RawGeometry::Circle { .. } => extract_circle(entity),
        RawGeometry::Arc { .. } => extract_arc(entity),
        RawGeometry::Ellipse { .. } => extract_ellipse(entity),
        _ => Err(BasicError::NotBasic),
    }
}
