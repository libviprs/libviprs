//! Curves cut exactly against a viewport clip (issue #1567, the geometry half of slice 6, #1565).
//!
//! The print profile draws nothing outside a viewport (P-7 of `docs/adr/pvec-print-profile.md`),
//! and PVec has no clip region, so the producer cuts. The design comment on #1557 settled how: a
//! curve that crosses the clip is split, never tessellated, and the pieces inside are kept.
//!
//! | source | pieces | class |
//! |---|---|---|
//! | circle, ellipse, arc (a trimmed conic) | trimmed arcs of the source's own carrier over the kept parameter intervals | `SupportedExact` |
//! | line, trimmed line, polyline | trimmed lines of the same carrier, polylines of the kept runs of the source's vertices | `SupportedExact` |
//! | B-spline, Bezier | sub-ranges of the source (`BSpline::sub_range`, PVec's knot insertion), the kind kept | `SupportedEquivalent` |
//! | trimmed spline | the same spline under a narrower trim | `SupportedEquivalent` |
//! | composite | each segment cut as above, the runs that stay joined kept as composites | the worst of its segments |
//! | rational B-spline | refused, [`CurveClipError::Unsupported`]: PVec has no sub-range for it yet | |
//!
//! # How
//!
//! Every crossing of the curve with an edge of the clip is found as a parameter of the curve. The
//! parameters cut the domain into intervals, and each interval is kept when the point PVec
//! evaluates at its middle is inside the clip by the even-odd rule. Kept neighbours are joined,
//! and for a whole circle or ellipse the interval through the seam of the parameter is one arc.
//! So a rectangle is just a four-vertex polygon here, and a polygon clip is cut edge by edge.
//!
//! A conic crossing is closed form: the edge is mapped into the frame where the conic is the unit
//! circle, one `sqrt` solves the quadratic, and the angle comes from `detmath::atan2`, so the bits
//! are the same on every machine and the angle is within the one ulp PVec states for it. A line or
//! polyline crossing is two cross products. A spline crossing has no closed form: each knot span is
//! a Bezier piece (PVec's `sub_range`), a piece whose control points all sit on one side of the
//! edge's line has no crossing (the convex hull), a piece whose control values change sign once has
//! exactly one and is bisected on PVec's evaluator, and anything else is halved again through
//! `sub_range`. The bisection stops at `Policy::V1`'s tolerance for the domain, which is reported
//! as the parameter tolerance with `SupportedEquivalent`.
//!
//! Nothing here evaluates, inserts knots or tessellates (NG-10 of
//! `docs/adr/pvec-normalized-geometry.md`): every point comes from `Curve2d::eval`, every spline
//! piece from `BSpline::sub_range`, and every tolerance from `Policy::V1`. A spurious crossing
//! parameter costs nothing, because neighbours on the same side are joined again, so an edge end
//! is matched with the policy's slack rather than missed.

use std::fmt;

use pvec_geometry::{Curve, Curve2d, CurveKind, GeometryError, Point2};

use super::fidelity::FidelityClass;

/// What became of one curve against one clip.
#[derive(Debug, Clone, PartialEq)]
pub enum CurveClip {
    /// Wholly inside: keep the source as it is.
    Inside,
    /// Wholly outside: nothing of it is drawn (`OutsideViewport`).
    Outside,
    /// It crosses the clip: the pieces inside, in the order the source runs through them.
    Split {
        /// The kept pieces.
        pieces: Vec<Curve>,
        /// `SupportedExact` for conics, lines and polylines, `SupportedEquivalent` once a spline
        /// is involved.
        fidelity: FidelityClass,
        /// How far, in the source parameter, a spline crossing may sit from the true one. Zero
        /// when every crossing is closed form.
        parameter_tolerance: f64,
    },
}

/// Why a curve could not be cut.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum CurveClipError {
    /// The clip has fewer than three vertices.
    Ring,
    /// A curve kind this module cannot cut exactly yet.
    Unsupported(CurveKind),
    /// PVec refused a piece.
    Geometry(GeometryError),
}

impl From<GeometryError> for CurveClipError {
    fn from(e: GeometryError) -> Self {
        CurveClipError::Geometry(e)
    }
}

impl fmt::Display for CurveClipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CurveClipError::Ring => write!(f, "a clip needs at least three vertices"),
            CurveClipError::Unsupported(k) => write!(f, "cannot cut a {k:?} exactly"),
            CurveClipError::Geometry(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for CurveClipError {}

/// Cuts `curve` against the polygon `ring` (its vertices in order, closed implicitly, inside by the
/// even-odd rule). A rectangle is the four corners.
pub fn clip_curve(curve: &Curve, ring: &[Point2]) -> Result<CurveClip, CurveClipError> {
    // Not cut yet: every curve is refused until the clipper lands (#1567).
    let _ = ring;
    Err(CurveClipError::Unsupported(curve.kind()))
}
