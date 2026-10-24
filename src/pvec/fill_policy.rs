//! The hatch and fill policy (N5.6, #1085). The decisions and their reasons are written up in
//! `docs/pvec-hatch-policy.md`, and this module is the same table as code, plus the counting
//! that puts every fill construct into the report.
//!
//! A fill is area display semantics. Nothing here turns a filled region into triangles or a
//! point set: a fill either leaves the geometry it already has (the subpaths N5.2 extracts) or
//! is counted and reported as not carried. Two things are deliberately not here yet. No fill
//! component is emitted, because the component stream (N6.6) is not built, so the PDF rows
//! decide "geometry only" and the winding rule stays on [`PdfPaintIntent`]. And no DWG hatch
//! boundary is kept: the provider maps neither `HATCH` nor `SOLID`, so both arrive as
//! [`RawGeometry::Unmapped`] and are counted here.

use std::collections::BTreeMap;

use super::fidelity::FidelityClass;
use super::pdf_vector::{PdfFillRule, PdfVectorPage};
use super::provider::{RawEntity, RawGeometry};

/// A source construct that paints or defines an area.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum FillConstruct {
    /// DWG `HATCH`, solid or pattern (the provider does not yet tell them apart).
    DwgHatch,
    /// DWG `SOLID`.
    DwgSolid,
    /// DWG `TRACE`.
    DwgTrace,
    /// A PDF fill operator (`f F f* B B* b b*`) under the nonzero rule.
    PdfFillNonZero,
    /// A PDF fill operator under the even-odd rule.
    PdfFillEvenOdd,
    /// A PDF `sh` shading paint.
    PdfShading,
    /// A PDF colour operator that selected a named pattern (tiling or shading).
    PdfPattern,
}

impl FillConstruct {
    /// Every construct, in a fixed order.
    pub const ALL: [FillConstruct; 7] = [
        FillConstruct::DwgHatch,
        FillConstruct::DwgSolid,
        FillConstruct::DwgTrace,
        FillConstruct::PdfFillNonZero,
        FillConstruct::PdfFillEvenOdd,
        FillConstruct::PdfShading,
        FillConstruct::PdfPattern,
    ];
}

/// What a construct becomes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FillDecision {
    /// The subpaths stay as geometry, the fill itself is not carried. The winding rule is
    /// recorded on the source side ([`PdfPaintIntent`](super::pdf_vector::PdfPaintIntent)).
    GeometryOnly,
    /// Not carried. Counted and reported with a reason.
    IgnoredCounted,
}

/// One recorded decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FillPolicyRow {
    /// The construct.
    pub construct: FillConstruct,
    /// What it becomes.
    pub decision: FillDecision,
    /// The report bucket the construct lands in when it is counted.
    pub bucket: FidelityClass,
    /// The stable reason the report carries.
    pub reason: &'static str,
}

/// The decisions, one row per construct.
pub const FILL_POLICY: [FillPolicyRow; 7] = [
    FillPolicyRow {
        construct: FillConstruct::DwgHatch,
        decision: FillDecision::IgnoredCounted,
        bucket: FidelityClass::IgnoredByPolicy,
        reason: "fill-hatch-not-in-v1",
    },
    FillPolicyRow {
        construct: FillConstruct::DwgSolid,
        decision: FillDecision::IgnoredCounted,
        bucket: FidelityClass::IgnoredByPolicy,
        reason: "fill-solid-not-in-v1",
    },
    FillPolicyRow {
        construct: FillConstruct::DwgTrace,
        decision: FillDecision::IgnoredCounted,
        bucket: FidelityClass::IgnoredByPolicy,
        reason: "fill-trace-not-in-v1",
    },
    FillPolicyRow {
        construct: FillConstruct::PdfFillNonZero,
        decision: FillDecision::GeometryOnly,
        bucket: FidelityClass::IgnoredByPolicy,
        reason: "fill-geometry-only-nonzero",
    },
    FillPolicyRow {
        construct: FillConstruct::PdfFillEvenOdd,
        decision: FillDecision::GeometryOnly,
        bucket: FidelityClass::IgnoredByPolicy,
        reason: "fill-geometry-only-even-odd",
    },
    FillPolicyRow {
        construct: FillConstruct::PdfShading,
        decision: FillDecision::IgnoredCounted,
        bucket: FidelityClass::RequiresExtension,
        reason: "fill-shading-requires-extension",
    },
    FillPolicyRow {
        construct: FillConstruct::PdfPattern,
        decision: FillDecision::IgnoredCounted,
        bucket: FidelityClass::RequiresExtension,
        reason: "fill-pattern-requires-extension",
    },
];

/// The row for a construct.
#[must_use]
pub fn policy_for(construct: FillConstruct) -> &'static FillPolicyRow {
    FILL_POLICY
        .iter()
        .find(|r| r.construct == construct)
        .expect("FILL_POLICY has a row for every FillConstruct")
}

/// Fill constructs counted so far.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FillReport {
    counts: BTreeMap<FillConstruct, u64>,
}

/// One report line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FillReportRow {
    /// The construct.
    pub construct: FillConstruct,
    /// How many were seen.
    pub count: u64,
    /// The bucket and reason from the policy.
    pub policy: FillPolicyRow,
}

impl FillReport {
    /// An empty report.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn add(&mut self, construct: FillConstruct, n: u64) {
        if n > 0 {
            *self.counts.entry(construct).or_insert(0) += n;
        }
    }

    /// How many of `construct` were seen.
    #[must_use]
    pub fn count(&self, construct: FillConstruct) -> u64 {
        self.counts.get(&construct).copied().unwrap_or(0)
    }

    /// Counts a PDF page: every filling entity by its winding rule, plus the page's shading
    /// and pattern counters. An entity that fills and strokes (`B`, `b`) is one fill here,
    /// its stroke is ordinary line geometry.
    pub fn count_pdf_page(&mut self, page: &PdfVectorPage) {
        for e in &page.entities {
            match e.paint.fill {
                Some(PdfFillRule::NonZero) => self.add(FillConstruct::PdfFillNonZero, 1),
                Some(PdfFillRule::EvenOdd) => self.add(FillConstruct::PdfFillEvenOdd, 1),
                _ => {}
            }
        }
        self.add(FillConstruct::PdfShading, u64::from(page.shading_paints));
        self.add(
            FillConstruct::PdfPattern,
            u64::from(page.pattern_selections),
        );
    }

    /// Counts DWG entities the provider did not map, by class name.
    pub fn count_dwg_entities<'a>(&mut self, entities: impl IntoIterator<Item = &'a RawEntity>) {
        for e in entities {
            if !matches!(e.geometry, RawGeometry::Unmapped { .. }) {
                continue;
            }
            match e.provenance.class_name.to_ascii_uppercase().as_str() {
                "HATCH" => self.add(FillConstruct::DwgHatch, 1),
                "SOLID" => self.add(FillConstruct::DwgSolid, 1),
                "TRACE" => self.add(FillConstruct::DwgTrace, 1),
                _ => {}
            }
        }
    }

    /// The non-zero rows in construct order, each with its policy.
    pub fn rows(&self) -> impl Iterator<Item = FillReportRow> + '_ {
        self.counts.iter().map(|(c, n)| FillReportRow {
            construct: *c,
            count: *n,
            policy: *policy_for(*c),
        })
    }
}
