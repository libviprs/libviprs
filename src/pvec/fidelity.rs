//! The six-way entity classification (N3.5, #1100).
//!
//! Every entity the ingestion meets ends in exactly one [`FidelityClass`], and "quietly dropped"
//! is not one of them. This module is the single source for that set: the conversion report
//! (N6.7), the classification-stability gate (N2.8) and the capability policy (N7.2) all read it,
//! and no second enum describing the same thing exists in the crate. The `Requires` enum that
//! `dwg_spline` used to carry locally is folded into it: what a refused spline needs is now a
//! [`ReasonCode`] whose class says whether the provider or PVec has to grow.
//!
//! # Shape
//!
//! * [`FidelityClass`] is the six classes.
//! * [`ReasonCode`] is the stable, machine-readable reason, enumerated in [`ReasonCode::ALL`]. Each
//!   code belongs to exactly one refusing class, so a code and its class cannot disagree.
//! * [`Classification`] is one entity's outcome: class, reason (for the refusing classes), the
//!   source class name, and the policy field for `IgnoredByPolicy`. Its constructors are the only
//!   way to build one, so a `Unsupported` without a reason is not representable.
//! * [`Tally`] counts classifications and [`Tally::finish`] checks the counts against the number
//!   of entities the caller saw arrive, which is the invariant that makes a dropped entity a
//!   conversion error instead of a silent loss.
//!
//! # Refusal, not tessellation
//!
//! `Unsupported` and `RequiresPVecCapability` produce no geometry (see
//! [`FidelityClass::produces_geometry`]), and `Normalized2DGeometry` refuses to hold a curve under
//! them. A curve that cannot be carried is reported, never flattened into a canonical PVec value.
//!
//! # What is not here
//!
//! Report rendering (N6.7), the caller-facing capability policy (N7.2) and the PVec capabilities
//! themselves (EPIC P). `IgnoredByPolicy` takes the responsible field as a name; the view-profile
//! API that owns those fields is N3.3, and it passes its field names in.

use std::collections::BTreeMap;
use std::fmt;

use super::provider::RawEntity;

/// What became of a source entity (NG-07): the six classes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum FidelityClass {
    /// The PVec value is the source construct, structurally.
    SupportedExact,
    /// The PVec value is the same locus under a different but equivalent construct. Exactness is
    /// claimed only through EPIC P's conformance classes.
    SupportedEquivalent,
    /// The mapping needs mathematics PVec has no entry point for yet. No geometry.
    RequiresPVecCapability,
    /// The geometry exists and needs an extension to be carried in full.
    RequiresExtension,
    /// The source construct has no mapping. No geometry.
    Unsupported,
    /// Dropped on purpose by policy. No geometry.
    IgnoredByPolicy,
}

impl FidelityClass {
    /// All six, in report order.
    pub const ALL: [FidelityClass; 6] = [
        FidelityClass::SupportedExact,
        FidelityClass::SupportedEquivalent,
        FidelityClass::RequiresPVecCapability,
        FidelityClass::RequiresExtension,
        FidelityClass::Unsupported,
        FidelityClass::IgnoredByPolicy,
    ];

    /// The class as the report spells it.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            FidelityClass::SupportedExact => "SupportedExact",
            FidelityClass::SupportedEquivalent => "SupportedEquivalent",
            FidelityClass::RequiresPVecCapability => "RequiresPVecCapability",
            FidelityClass::RequiresExtension => "RequiresExtension",
            FidelityClass::Unsupported => "Unsupported",
            FidelityClass::IgnoredByPolicy => "IgnoredByPolicy",
        }
    }

    /// Whether a record of this class carries a PVec curve (NG-07).
    #[must_use]
    pub const fn produces_geometry(self) -> bool {
        matches!(
            self,
            FidelityClass::SupportedExact
                | FidelityClass::SupportedEquivalent
                | FidelityClass::RequiresExtension
        )
    }

    const fn index(self) -> usize {
        match self {
            FidelityClass::SupportedExact => 0,
            FidelityClass::SupportedEquivalent => 1,
            FidelityClass::RequiresPVecCapability => 2,
            FidelityClass::RequiresExtension => 3,
            FidelityClass::Unsupported => 4,
            FidelityClass::IgnoredByPolicy => 5,
        }
    }
}

/// Why an entity is not supported, as a stable code. The whole set is [`ReasonCode::ALL`].
///
/// The spellings from [`ReasonCode::code`] are what the baseline diff keys on, so renaming one is
/// a deliberate change to that baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum ReasonCode {
    /// A spline's knot vector is missing or the provider cannot report it. `RequiresExtension`.
    SplineKnotsNotReported,
    /// A spline's periodic flag is missing or not reportable. `RequiresExtension`.
    SplinePeriodicNotReported,
    /// A spline's weights are not reportable, so rational cannot be told from not. `RequiresExtension`.
    SplineWeightsNotReported,
    /// A spline with control points off one plane: PVec's curves are planar.
    /// `RequiresPVecCapability`.
    Spline3d,
    /// A `HELIX`: PVec's curves are planar. `RequiresPVecCapability`.
    Helix,
    /// A spline beyond PVec's degree or size limits. `Unsupported`.
    SplineLimitExceeded,
    /// A spline whose knots, weights or coordinates are inconsistent or not finite. `Unsupported`.
    SplineMalformed,
    /// A spline PVec's numeric policy refused (repeat rules, domain, weight floor). `Unsupported`.
    SplinePolicyRefused,
    /// A source class the provider saw and does not map. `Unsupported`.
    UnmappedSourceClass,
    /// The only available representation is for display, not the entity's own geometry.
    /// `Unsupported`.
    DisplayFidelityOnly,
    /// A PDF soft mask is in force: it modulates what is painted, and PVec has no
    /// construct for it (N5.3). `Unsupported`.
    PdfSoftMask,
}

impl ReasonCode {
    /// Every code, in a fixed order.
    pub const ALL: [ReasonCode; 11] = [
        ReasonCode::SplineKnotsNotReported,
        ReasonCode::SplinePeriodicNotReported,
        ReasonCode::SplineWeightsNotReported,
        ReasonCode::Spline3d,
        ReasonCode::Helix,
        ReasonCode::SplineLimitExceeded,
        ReasonCode::SplineMalformed,
        ReasonCode::SplinePolicyRefused,
        ReasonCode::UnmappedSourceClass,
        ReasonCode::DisplayFidelityOnly,
        ReasonCode::PdfSoftMask,
    ];

    /// The stable machine-readable spelling.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            ReasonCode::SplineKnotsNotReported => "spline-knots-not-reported",
            ReasonCode::SplinePeriodicNotReported => "spline-periodic-not-reported",
            ReasonCode::SplineWeightsNotReported => "spline-weights-not-reported",
            ReasonCode::Spline3d => "spline-3d",
            ReasonCode::Helix => "helix",
            ReasonCode::SplineLimitExceeded => "spline-limit-exceeded",
            ReasonCode::SplineMalformed => "spline-malformed",
            ReasonCode::SplinePolicyRefused => "spline-policy-refused",
            ReasonCode::UnmappedSourceClass => "unmapped-source-class",
            ReasonCode::DisplayFidelityOnly => "display-fidelity-only",
            ReasonCode::PdfSoftMask => "pdf-soft-mask",
        }
    }

    /// The class this reason puts an entity in.
    #[must_use]
    pub const fn class(self) -> FidelityClass {
        match self {
            ReasonCode::SplineKnotsNotReported
            | ReasonCode::SplinePeriodicNotReported
            | ReasonCode::SplineWeightsNotReported => FidelityClass::RequiresExtension,
            ReasonCode::Spline3d | ReasonCode::Helix => FidelityClass::RequiresPVecCapability,
            ReasonCode::SplineLimitExceeded
            | ReasonCode::SplineMalformed
            | ReasonCode::SplinePolicyRefused
            | ReasonCode::UnmappedSourceClass
            | ReasonCode::DisplayFidelityOnly
            | ReasonCode::PdfSoftMask => FidelityClass::Unsupported,
        }
    }
}

impl fmt::Display for ReasonCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

/// How faithfully a supported mapping holds the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Basis {
    /// The structural parameters are preserved.
    Structural,
    /// A mathematically equivalent construct.
    Equivalent,
    /// Only a display rendering exists. Never exact.
    DisplayOnly,
}

/// One entity's outcome. Built only through the constructors below.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Classification {
    class: FidelityClass,
    reason: Option<ReasonCode>,
    source_class: String,
    policy_field: Option<String>,
}

impl Classification {
    /// A supported mapping. [`Basis::DisplayOnly`] is never `SupportedExact`: it comes back
    /// `Unsupported` with [`ReasonCode::DisplayFidelityOnly`].
    #[must_use]
    pub fn supported(basis: Basis, source_class: impl Into<String>) -> Self {
        match basis {
            Basis::Structural => Self::plain(FidelityClass::SupportedExact, source_class),
            Basis::Equivalent => Self::plain(FidelityClass::SupportedEquivalent, source_class),
            Basis::DisplayOnly => Self::refused(ReasonCode::DisplayFidelityOnly, source_class),
        }
    }

    /// A refusal: the class is the reason's own.
    #[must_use]
    pub fn refused(reason: ReasonCode, source_class: impl Into<String>) -> Self {
        Classification {
            class: reason.class(),
            reason: Some(reason),
            source_class: source_class.into(),
            policy_field: None,
        }
    }

    /// Excluded by the active view profile, naming the field responsible.
    #[must_use]
    pub fn ignored_by_policy(field: impl Into<String>, source_class: impl Into<String>) -> Self {
        Classification {
            class: FidelityClass::IgnoredByPolicy,
            reason: None,
            source_class: source_class.into(),
            policy_field: Some(field.into()),
        }
    }

    fn plain(class: FidelityClass, source_class: impl Into<String>) -> Self {
        Classification {
            class,
            reason: None,
            source_class: source_class.into(),
            policy_field: None,
        }
    }

    /// The class.
    #[must_use]
    pub fn class(&self) -> FidelityClass {
        self.class
    }

    /// The reason code, present for every refusing class.
    #[must_use]
    pub fn reason(&self) -> Option<ReasonCode> {
        self.reason
    }

    /// The drawing's class name for the entity (`SPLINE`, `HELIX`).
    #[must_use]
    pub fn source_class(&self) -> &str {
        &self.source_class
    }

    /// The view-profile field that excluded the entity, for `IgnoredByPolicy`.
    #[must_use]
    pub fn policy_field(&self) -> Option<&str> {
        self.policy_field.as_deref()
    }
}

/// Classifies an entity the provider saw and does not map (`RawGeometry::Unmapped`). It is
/// `Unsupported` with [`ReasonCode::UnmappedSourceClass`], counted and never dropped.
#[must_use]
pub fn classify_unmapped(entity: &RawEntity) -> Classification {
    Classification::refused(
        ReasonCode::UnmappedSourceClass,
        entity.provenance.class_name.as_str(),
    )
}

/// The counts disagree with the entities encountered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TallyError {
    /// An entity was dropped (or counted twice): the classified count is not the encountered one.
    CountMismatch {
        /// How many classifications were recorded.
        classified: u64,
        /// How many entities the conversion saw arrive.
        encountered: u64,
    },
}

impl fmt::Display for TallyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TallyError::CountMismatch {
                classified,
                encountered,
            } => write!(
                f,
                "{classified} classifications for {encountered} encountered entities"
            ),
        }
    }
}

impl std::error::Error for TallyError {}

/// Counts classifications as a conversion makes them.
#[derive(Debug, Clone, Default)]
pub struct Tally {
    counts: [u64; 6],
    reasons: BTreeMap<(ReasonCode, String), u64>,
}

impl Tally {
    /// An empty tally.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Records one entity's classification.
    pub fn record(&mut self, c: &Classification) {
        self.counts[c.class.index()] += 1;
        if let Some(r) = c.reason {
            *self.reasons.entry((r, c.source_class.clone())).or_insert(0) += 1;
        }
    }

    /// Closes the tally against the number of entities the conversion saw arrive, counted
    /// independently of the classifications.
    ///
    /// # Errors
    /// [`TallyError::CountMismatch`] when the six counts do not sum to `encountered`.
    pub fn finish(self, encountered: u64) -> Result<FidelityReport, TallyError> {
        let classified: u64 = self.counts.iter().sum();
        if classified != encountered {
            return Err(TallyError::CountMismatch {
                classified,
                encountered,
            });
        }
        Ok(FidelityReport {
            counts: self.counts,
            reasons: self.reasons,
            encountered,
        })
    }
}

/// A closed tally: the six counts, known to sum to the entities encountered.
#[derive(Debug, Clone)]
pub struct FidelityReport {
    counts: [u64; 6],
    reasons: BTreeMap<(ReasonCode, String), u64>,
    encountered: u64,
}

impl FidelityReport {
    /// Entities in `class`.
    #[must_use]
    pub fn count(&self, class: FidelityClass) -> u64 {
        self.counts[class.index()]
    }

    /// Entities encountered, equal to the sum of the six counts.
    #[must_use]
    pub fn encountered(&self) -> u64 {
        self.encountered
    }

    /// `(reason, source class, count)` rows in a fixed order, so a baseline diff is meaningful.
    pub fn reasons(&self) -> impl Iterator<Item = (ReasonCode, &str, u64)> {
        self.reasons.iter().map(|((r, s), n)| (*r, s.as_str(), *n))
    }
}
