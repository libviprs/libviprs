//! PDF path segments into PVec's semantic types, curves kept as curves (N5.2, #1081).
//!
//! A PDF path is already exact mathematics: `c`, `v` and `y` are cubic Beziers with `f64`
//! control points. Flattening them on ingest because a renderer will flatten them anyway would
//! throw away the analytic form the integration epic exists to keep, so this module does the
//! opposite and hands every control point to PVec as it came out of the content stream.
//!
//! ```text
//! PdfPathEntity.segments          PVec semantic type                  class
//!   c  (v, y already expanded)    Curve::Bezier, degree 3, 4 points   SupportedExact
//!   l                             Curve::Polyline, two vertices       SupportedEquivalent
//!   h                             the closing Polyline, only when the
//!                                 pen is not already at the start     SupportedEquivalent
//!   re (m, 3 l, h upstream)       four Polylines, subpath closed      SupportedEquivalent
//! ```
//!
//! # Why a line is a two-vertex polyline
//!
//! PVec has no segment constructor from two end points (ADR NG-11): its `Line` is unbounded and a
//! trimmed line would need a unit direction, which is arithmetic on the coordinates. A polyline
//! with two vertices is the same locus built without any, so it is `SupportedEquivalent` and
//! never `SupportedExact`. Where PVec grows the entry point, this is the one place to change.
//!
//! # One entity per segment, closure per subpath
//!
//! Each drawn segment is its own entity, in content-stream order, so an S-curve of two `c`
//! operators is two curve entities. Closure is not a geometry: it is a flag on a
//! [`PdfSubpath`], and `h` adds a closing segment (never a degenerate curve) only when the pen
//! is not already back at the start. A compound path (several subpaths painted by one operator)
//! stays one [`PdfNormalizedPath`] whose entities all carry the same
//! [`compound_group`](PdfNormalizedEntity::compound_group), the id the provenance stream reports
//! for them. They are never merged into one entity.
//!
//! # Refusal, not approximation
//!
//! If PVec refuses a segment (a zero-length line, a curve whose hull is out of range), the
//! entity is kept, classified `Unsupported` with the reason and PVec's own error attached. There
//! is no sampling, no arc fitting and no polyline fallback anywhere in this module, and no
//! function here takes a tolerance, a step count or a segment budget.
//!
//! # Claim
//!
//! A cubic that round-trips here is geometrically exact with respect to the PDF. That says
//! nothing about the DWG the PDF may have been plotted from; that comparison is N5.7's. The
//! three conformance class names are EPIC P's and are only reused.

use pvec_geometry::{Bezier, Curve, Point2, Polyline};

use super::fidelity::{Basis, Classification, FidelityClass, ReasonCode};
use super::normalized::{NormalizeError, Normalized2DGeometry};
use super::pdf_vector::{PdfPathEntity, PdfPathSegment, PdfPoint, PdfProvenance, PdfVectorPage};

/// One drawn segment of a PDF path, as a PVec value or as a counted refusal.
#[derive(Debug, Clone)]
pub struct PdfNormalizedEntity {
    /// The PVec record. Holds no curve when PVec refused the segment.
    pub geometry: Normalized2DGeometry,
    /// What became of the segment, with a reason code when it was refused.
    pub classification: Classification,
    /// Index into [`PdfNormalizedPath::subpaths`] of the subpath this segment belongs to.
    pub subpath: usize,
    /// Shared by every entity of one painted path (a compound path).
    pub compound_group: u64,
    /// PVec's refusal, unchanged, when `geometry` holds no curve.
    pub refusal: Option<NormalizeError>,
}

/// The extent and closure of one subpath inside [`PdfNormalizedPath::entities`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PdfSubpath {
    /// Whether the subpath was closed with `h` or `re`.
    pub closed: bool,
    /// Index of its first entity.
    pub first_entity: usize,
    /// How many entities it holds.
    pub entity_count: usize,
}

/// One painted path, normalized.
#[derive(Debug, Clone)]
pub struct PdfNormalizedPath {
    /// The compound-path group id, the same value on every entity.
    pub compound_group: u64,
    /// Entities in content-stream order.
    pub entities: Vec<PdfNormalizedEntity>,
    /// The subpaths, each with its own closure state.
    pub subpaths: Vec<PdfSubpath>,
    /// Where the path came from, as N5.1 recorded it.
    pub provenance: PdfProvenance,
}

fn same(a: PdfPoint, b: PdfPoint) -> bool {
    a[0].to_bits() == b[0].to_bits() && a[1].to_bits() == b[1].to_bits()
}

fn point(p: PdfPoint, field: &'static str) -> Result<Point2, NormalizeError> {
    Point2::checked(p[0], p[1]).map_err(|error| NormalizeError::Refused { field, error })
}

fn cubic(pts: [PdfPoint; 4]) -> Result<Curve, NormalizeError> {
    let mut cps = Vec::with_capacity(4);
    for p in pts {
        cps.push(point(p, "control_points")?);
    }
    Bezier::new(cps)
        .map(Curve::Bezier)
        .map_err(|error| NormalizeError::Refused {
            field: "control_points",
            error,
        })
}

fn segment(a: PdfPoint, b: PdfPoint) -> Result<Curve, NormalizeError> {
    let vertices = vec![point(a, "vertices")?, point(b, "vertices")?];
    Polyline::new(vertices)
        .map(Curve::Polyline)
        .map_err(|error| NormalizeError::Refused {
            field: "vertices",
            error,
        })
}

fn entity(
    built: Result<Curve, NormalizeError>,
    basis: Basis,
    source_class: &str,
    subpath: usize,
    compound_group: u64,
) -> PdfNormalizedEntity {
    let accepted = built.and_then(|curve| {
        let class = Classification::supported(basis, source_class);
        Normalized2DGeometry::from_curve(curve, class.class()).map(|g| (g, class))
    });
    match accepted {
        Ok((geometry, classification)) => PdfNormalizedEntity {
            geometry,
            classification,
            subpath,
            compound_group,
            refusal: None,
        },
        Err(refusal) => PdfNormalizedEntity {
            // `Unsupported` carries no geometry, so this cannot itself be refused.
            geometry: Normalized2DGeometry::without_geometry(FidelityClass::Unsupported)
                .expect("Unsupported produces no geometry"),
            classification: Classification::refused(ReasonCode::SplinePolicyRefused, source_class),
            subpath,
            compound_group,
            refusal: Some(refusal),
        },
    }
}

/// Maps one extracted path. Never fails: a segment PVec refuses is kept as a classified entity.
#[must_use]
pub fn normalize_path(path: &PdfPathEntity) -> PdfNormalizedPath {
    let group = path.provenance.sequence;
    let mut entities: Vec<PdfNormalizedEntity> = Vec::new();
    let mut subpaths: Vec<PdfSubpath> = Vec::new();
    // The pen, the subpath start, and where the open subpath's entities begin.
    let mut pen: Option<PdfPoint> = None;
    let mut start: Option<PdfPoint> = None;
    let mut open_from: Option<usize> = None;

    fn close_subpath(
        subpaths: &mut Vec<PdfSubpath>,
        entities: &[PdfNormalizedEntity],
        open_from: &mut Option<usize>,
        closed: bool,
    ) {
        if let Some(first) = open_from.take() {
            let count = entities.len() - first;
            if count > 0 {
                subpaths.push(PdfSubpath {
                    closed,
                    first_entity: first,
                    entity_count: count,
                });
            }
        }
    }

    for seg in &path.segments {
        match *seg {
            PdfPathSegment::MoveTo(p) => {
                close_subpath(&mut subpaths, &entities, &mut open_from, false);
                pen = Some(p);
                start = Some(p);
            }
            PdfPathSegment::LineTo(p) => {
                let from = pen.unwrap_or(p);
                open_from.get_or_insert(entities.len());
                entities.push(entity(
                    segment(from, p),
                    Basis::Equivalent,
                    "PDF l",
                    subpaths.len(),
                    group,
                ));
                start.get_or_insert(from);
                pen = Some(p);
            }
            PdfPathSegment::CurveTo { c1, c2, end } => {
                let from = pen.unwrap_or(c1);
                open_from.get_or_insert(entities.len());
                entities.push(entity(
                    cubic([from, c1, c2, end]),
                    Basis::Structural,
                    "PDF c",
                    subpaths.len(),
                    group,
                ));
                start.get_or_insert(from);
                pen = Some(end);
            }
            PdfPathSegment::Close => {
                if let (Some(from), Some(to)) = (pen, start) {
                    if open_from.is_some() && !same(from, to) {
                        entities.push(entity(
                            segment(from, to),
                            Basis::Equivalent,
                            "PDF h",
                            subpaths.len(),
                            group,
                        ));
                    }
                    close_subpath(&mut subpaths, &entities, &mut open_from, true);
                    pen = Some(to);
                }
            }
        }
    }
    close_subpath(&mut subpaths, &entities, &mut open_from, false);

    PdfNormalizedPath {
        compound_group: group,
        entities,
        subpaths,
        provenance: path.provenance.clone(),
    }
}

/// Maps every path of a page, in page order.
#[must_use]
pub fn normalize_page(page: &PdfVectorPage) -> Vec<PdfNormalizedPath> {
    page.entities.iter().map(normalize_path).collect()
}
