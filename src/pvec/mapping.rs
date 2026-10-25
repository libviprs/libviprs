//! The one table that maps normalized geometry to PVec (N6.2, issue #1089).
//!
//! Every kind of normalized geometry has exactly one row in [`NORMALIZED_TO_PVEC`]: the PVec
//! semantic type it becomes, the conformance class that mapping claims, the PVec capability it
//! needs, and what the record is classified as when PVec does not have that capability.
//!
//! The last cell is an [`AbsentCapability`], a type with three values. Tessellation is not one
//! of them, so a curve that PVec cannot carry is reported as such and never quietly turned into
//! a polyline. `tests/pvec_1089_mapping.rs` asserts that over the whole table.
//!
//! Adding a [`NormalizedKind`] fails to compile until [`row_index`] (an exhaustive `match`, no
//! wildcard arm) gives it a row. `docs/pvec-mapping.md` is [`render_markdown`] written to disk
//! and a test compares the two byte for byte, so the doc cannot drift from the table.
//!
//! The capability ids are PVec's registry ids (`docs/CAPABILITIES.md` in `libviprs/pvec`),
//! recorded here as data. Nothing in this module evaluates, splits or samples a curve.

use std::fmt::Write as _;

use pvec_geometry::Curve;

use super::fidelity::FidelityClass;
use super::normalized::Normalized2DGeometry;

/// The kinds of normalized geometry that carry a PVec curve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum NormalizedKind {
    /// A line segment, a trimmed PVec line.
    Line,
    /// A full circle.
    Circle,
    /// A circular arc, a trimmed circle.
    CircularArc,
    /// A full ellipse.
    Ellipse,
    /// An elliptical arc, a trimmed ellipse.
    EllipticalArc,
    /// A polyline of straight segments.
    Polyline,
    /// A chain of arcs and straight runs (a bulge polyline).
    Composite,
    /// A Bezier curve.
    Bezier,
    /// A non-rational B-spline with explicit knots.
    BSpline,
    /// A rational B-spline.
    RationalBSpline,
}

impl NormalizedKind {
    /// Every kind, in table order.
    pub const ALL: [NormalizedKind; 10] = [
        NormalizedKind::Line,
        NormalizedKind::Circle,
        NormalizedKind::CircularArc,
        NormalizedKind::Ellipse,
        NormalizedKind::EllipticalArc,
        NormalizedKind::Polyline,
        NormalizedKind::Composite,
        NormalizedKind::Bezier,
        NormalizedKind::BSpline,
        NormalizedKind::RationalBSpline,
    ];

    /// The kind as the doc spells it.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            NormalizedKind::Line => "Line",
            NormalizedKind::Circle => "Circle",
            NormalizedKind::CircularArc => "CircularArc",
            NormalizedKind::Ellipse => "Ellipse",
            NormalizedKind::EllipticalArc => "EllipticalArc",
            NormalizedKind::Polyline => "Polyline",
            NormalizedKind::Composite => "Composite",
            NormalizedKind::Bezier => "Bezier",
            NormalizedKind::BSpline => "BSpline",
            NormalizedKind::RationalBSpline => "RationalBSpline",
        }
    }
}

/// EPIC P's three conformance classes: what a mapping is allowed to claim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConformanceClass {
    /// The PVec value is the source construct, parameter for parameter.
    StructuralExactness,
    /// The PVec value is the same locus under an equivalent construct.
    GeometricExactness,
    /// The PVec value is a densified display form. No row here uses it.
    DisplayFidelity,
}

impl ConformanceClass {
    /// The class as the doc spells it.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            ConformanceClass::StructuralExactness => "structural exactness",
            ConformanceClass::GeometricExactness => "geometric exactness",
            ConformanceClass::DisplayFidelity => "display fidelity",
        }
    }
}

/// What a record is classified as when the capability its row names is absent. Exactly the three
/// classes the issue allows, so tessellation is unrepresentable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbsentCapability {
    /// PVec has no entry point for it yet.
    RequiresPVecCapability,
    /// Carrying it in full needs an extension.
    RequiresExtension,
    /// There is no mapping.
    Unsupported,
}

impl AbsentCapability {
    /// All three.
    pub const ALL: [AbsentCapability; 3] = [
        AbsentCapability::RequiresPVecCapability,
        AbsentCapability::RequiresExtension,
        AbsentCapability::Unsupported,
    ];

    /// The N3.5 class a record gets.
    #[must_use]
    pub const fn fidelity(self) -> FidelityClass {
        match self {
            AbsentCapability::RequiresPVecCapability => FidelityClass::RequiresPVecCapability,
            AbsentCapability::RequiresExtension => FidelityClass::RequiresExtension,
            AbsentCapability::Unsupported => FidelityClass::Unsupported,
        }
    }
}

/// A PVec capability: its registry id and name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapabilityId {
    /// PVec's registry id. Never 0.
    pub id: u16,
    /// PVec's registry name.
    pub name: &'static str,
}

const TRIMMED_CURVE: CapabilityId = CapabilityId {
    id: 8,
    name: "trimmed_curve",
};

/// One row of the table.
#[derive(Debug, Clone, Copy)]
pub struct MappingRow {
    /// The normalized kind.
    pub kind: NormalizedKind,
    /// The PVec semantic type it becomes.
    pub pvec_type: &'static str,
    /// What the mapping claims.
    pub conformance: ConformanceClass,
    /// The PVec capability it needs.
    pub capability: CapabilityId,
    /// The classification when that capability is absent.
    pub absent: AbsentCapability,
}

impl MappingRow {
    /// The N3.5 class a record of this kind carries when the capability is present.
    #[must_use]
    pub const fn fidelity_when_present(&self) -> FidelityClass {
        match self.conformance {
            ConformanceClass::StructuralExactness => FidelityClass::SupportedExact,
            ConformanceClass::GeometricExactness => FidelityClass::SupportedEquivalent,
            // A display-only form is an extension to carry, never exact.
            ConformanceClass::DisplayFidelity => FidelityClass::RequiresExtension,
        }
    }
}

const fn row(
    kind: NormalizedKind,
    pvec_type: &'static str,
    conformance: ConformanceClass,
    capability: CapabilityId,
) -> MappingRow {
    MappingRow {
        kind,
        pvec_type,
        conformance,
        capability,
        absent: AbsentCapability::RequiresPVecCapability,
    }
}

use ConformanceClass::{GeometricExactness as GEO, StructuralExactness as STR};
use NormalizedKind as K;

/// The table, in [`NormalizedKind::ALL`] order.
pub const NORMALIZED_TO_PVEC: [MappingRow; 10] = [
    row(K::Line, "trimmed_curve(line)", STR, TRIMMED_CURVE),
    row(
        K::Circle,
        "circle",
        STR,
        CapabilityId {
            id: 6,
            name: "circle",
        },
    ),
    row(K::CircularArc, "trimmed_curve(circle)", STR, TRIMMED_CURVE),
    row(
        K::Ellipse,
        "ellipse",
        STR,
        CapabilityId {
            id: 7,
            name: "ellipse",
        },
    ),
    row(
        K::EllipticalArc,
        "trimmed_curve(ellipse)",
        STR,
        TRIMMED_CURVE,
    ),
    row(
        K::Polyline,
        "polyline",
        STR,
        CapabilityId {
            id: 5,
            name: "polyline",
        },
    ),
    row(
        K::Composite,
        "composite_curve",
        GEO,
        CapabilityId {
            id: 9,
            name: "composite_curve",
        },
    ),
    row(
        K::Bezier,
        "bezier_curve",
        STR,
        CapabilityId {
            id: 10,
            name: "bezier_curve",
        },
    ),
    row(
        K::BSpline,
        "b_spline_curve",
        STR,
        CapabilityId {
            id: 11,
            name: "b_spline_curve",
        },
    ),
    row(
        K::RationalBSpline,
        "rational_b_spline_curve",
        STR,
        CapabilityId {
            id: 12,
            name: "rational_b_spline_curve",
        },
    ),
];

/// Where a kind's row sits in [`NORMALIZED_TO_PVEC`]. An exhaustive `match` with no wildcard:
/// a new [`NormalizedKind`] does not compile until it is placed here, and the table test then
/// insists the row is really there.
#[must_use]
pub const fn row_index(kind: NormalizedKind) -> usize {
    match kind {
        NormalizedKind::Line => 0,
        NormalizedKind::Circle => 1,
        NormalizedKind::CircularArc => 2,
        NormalizedKind::Ellipse => 3,
        NormalizedKind::EllipticalArc => 4,
        NormalizedKind::Polyline => 5,
        NormalizedKind::Composite => 6,
        NormalizedKind::Bezier => 7,
        NormalizedKind::BSpline => 8,
        NormalizedKind::RationalBSpline => 9,
    }
}

/// The row for a kind.
#[must_use]
pub fn row_for(kind: NormalizedKind) -> &'static MappingRow {
    &NORMALIZED_TO_PVEC[row_index(kind)]
}

/// The kind of a record's curve, or `None` for a record with no geometry and for a curve shape
/// the table has no row for (the caller reports it, this does not guess).
#[must_use]
pub fn kind_of(geometry: &Normalized2DGeometry) -> Option<NormalizedKind> {
    match geometry.geometry()? {
        Curve::Circle(_) => Some(NormalizedKind::Circle),
        Curve::Ellipse(_) => Some(NormalizedKind::Ellipse),
        Curve::Polyline(_) => Some(NormalizedKind::Polyline),
        Curve::Composite(_) => Some(NormalizedKind::Composite),
        Curve::Bezier(_) => Some(NormalizedKind::Bezier),
        Curve::BSpline(_) => Some(NormalizedKind::BSpline),
        Curve::RationalBSpline(_) => Some(NormalizedKind::RationalBSpline),
        Curve::Trimmed(t) => match t.basis() {
            Curve::Line(_) => Some(NormalizedKind::Line),
            Curve::Circle(_) => Some(NormalizedKind::CircularArc),
            Curve::Ellipse(_) => Some(NormalizedKind::EllipticalArc),
            _ => None,
        },
        _ => None,
    }
}

/// `docs/pvec-mapping.md`, rendered from the table.
#[must_use]
pub fn render_markdown() -> String {
    let mut out = String::new();
    out.push_str(
        "# Normalized geometry to PVec (N6.2, issue #1089)\n\n\
         This file is generated from `NORMALIZED_TO_PVEC` in `src/pvec/mapping.rs`. Do not edit it by hand: \
         a test renders the table and fails if the committed file differs by a byte. To change it, \
         change the table and run \
         `PVEC_MAPPING_BLESS=1 cargo test --features pvec-normalized --test pvec_1089_mapping`.\n\n\
         The capability ids are PVec's registry ids. The last column is what a record is classified \
         as when PVec does not have that capability. Tessellation is never an answer, and no curve \
         row maps to a polyline or a point set.\n\n\
         | normalized | PVec type | conformance class | capability | when absent |\n\
         |---|---|---|---|---|\n",
    );
    for r in NORMALIZED_TO_PVEC.iter() {
        let _ = writeln!(
            out,
            "| `{}` | `{}` | {} | {} `{}` | `{}` |",
            r.kind.tag(),
            r.pvec_type,
            r.conformance.tag(),
            r.capability.id,
            r.capability.name,
            r.absent.fidelity().tag(),
        );
    }
    out
}
