//! Drift check (red-test stub: the API exists, the checks do not yet): a `.pvec` against the fixture it was generated from (issue #1471 follow-up, EPIC N).
//!
//! The artifact is read back through PVec's public reader and its geometry is compared, per entity
//! id, with what the fixture knows about itself:
//!
//! - a `KNOWN_TRUTH_EXACT` fixture is compared with its `expected.math.json`, the independently
//!   authored truth, so a pass here says the artifact is the drawing and not merely the same as last
//!   time;
//! - every other evidence class (`DUAL_ORACLE_EQUIVALENT`, `REAL_WORLD_DIFFERENTIAL`) has no truth,
//!   so the artifact is compared with the previously accepted output, a canonical dump kept in the
//!   repository;
//! - a `QUARANTINED` fixture is compared the same way, and the report says it does not gate.
//!
//! A known-truth fixture is also compared with its accepted dump when one is passed in, so the report
//! can say both "this is not the drawing" and "this is not what we accepted last time".
//!
//! Every comparison is within PVec's shared numeric policy ([`Policy::V1`]); there is no epsilon of
//! libviprs's own. The curve mathematics (evaluation, bounds, trims) is PVec's, reached through its
//! public API. What this module adds is the bookkeeping: which entity is which, and what kind of
//! drift a difference is.
//!
//! The canonical dump ([`canonical_dump`]) prints every number as its IEEE bits next to the shortest
//! decimal, so its sha256 ([`digest`]) changes when a single bit of the stored geometry does. That
//! digest is what `tests/fixtures/pvec-drift/` commits, one per fixture.
//!
//! See `docs/pvec-drift-check.md`.

use std::fmt::{self, Write as _};
use std::path::Path;

use serde_json::Value;

/// What kind of drift a difference is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DriftKind {
    /// A vertex, endpoint, centre, control point, knot or weight is not where it should be.
    MovedVertex,
    /// A radius or semi-axis changed.
    ChangedRadius,
    /// An entity the fixture has is not in the artifact.
    DroppedEntity,
    /// The artifact has an entity the fixture does not.
    ExtraEntity,
    /// The right geometry is under another entity id.
    ReorderedIds,
    /// The manifest bounds are not the bounds of the geometry.
    ChangedBounds,
    /// The manifest unit is not the fixture's.
    ChangedUnit,
    /// The entity is another kind of curve than the fixture says.
    SwappedClass,
}

impl DriftKind {
    /// The name used in the machine-readable report.
    pub fn name(self) -> &'static str {
        match self {
            Self::MovedVertex => "moved-vertex",
            Self::ChangedRadius => "changed-radius",
            Self::DroppedEntity => "dropped-entity",
            Self::ExtraEntity => "extra-entity",
            Self::ReorderedIds => "reordered-ids",
            Self::ChangedBounds => "changed-bounds",
            Self::ChangedUnit => "changed-unit",
            Self::SwappedClass => "swapped-class",
        }
    }
}

/// What a drift was found against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Against {
    /// The fixture's `expected.math.json`.
    Truth,
    /// The previously accepted output.
    Accepted,
}

impl Against {
    fn name(self) -> &'static str {
        match self {
            Self::Truth => "known-truth",
            Self::Accepted => "accepted-output",
        }
    }
}

/// One difference between the artifact and what it was checked against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Drift {
    /// The PVec entity id, when the difference belongs to one.
    pub entity: Option<u32>,
    /// What kind of difference.
    pub kind: DriftKind,
    /// What it was compared with.
    pub against: Against,
    /// The difference, in words.
    pub detail: String,
}

/// The result of checking one artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DriftReport {
    /// The fixture id.
    pub fixture: String,
    /// The fixture's evidence class, as the corpus names it.
    pub evidence_class: String,
    /// Whether a drift fails the check: false for a quarantined fixture.
    pub gating: bool,
    /// Every difference found, sorted by entity id and then kind.
    pub drifts: Vec<Drift>,
    /// sha256 of the canonical dump of the artifact.
    pub digest: String,
}

impl DriftReport {
    /// No drift was found.
    pub fn is_clean(&self) -> bool {
        self.drifts.is_empty()
    }

    /// The report fails the check: it has a drift and the fixture gates.
    pub fn fails(&self) -> bool {
        self.gating && !self.drifts.is_empty()
    }

    /// The machine-readable report: one JSON object, keys in a fixed order.
    pub fn to_json(&self) -> String {
        let drifts: Vec<Value> = self
            .drifts
            .iter()
            .map(|d| {
                serde_json::json!({
                    "entity": d.entity,
                    "kind": d.kind.name(),
                    "against": d.against.name(),
                    "detail": d.detail,
                })
            })
            .collect();
        let mut out = serde_json::to_string_pretty(&serde_json::json!({
            "fixture": self.fixture,
            "evidence_class": self.evidence_class,
            "gating": self.gating,
            "clean": self.is_clean(),
            "digest": self.digest,
            "drifts": drifts,
        }))
        .expect("a json value serialises");
        out.push('\n');
        out
    }

    /// The report for a person.
    pub fn to_text(&self) -> String {
        let mut out = format!(
            "{} ({}): {}\n",
            self.fixture,
            self.evidence_class,
            if self.is_clean() {
                "no drift".to_owned()
            } else {
                format!(
                    "{} drift(s){}",
                    self.drifts.len(),
                    if self.gating { "" } else { ", not gating" }
                )
            }
        );
        for d in &self.drifts {
            let who = d
                .entity
                .map_or_else(|| "artifact".to_owned(), |e| format!("entity {e}"));
            let _ = writeln!(
                out,
                "  {who}: {} against {}: {}",
                d.kind.name(),
                d.against.name(),
                d.detail
            );
        }
        let _ = writeln!(out, "  digest {}", self.digest);
        out
    }
}

/// Why a check could not be made (as opposed to a drift being found).
#[derive(Debug)]
pub enum DriftError {
    /// A file could not be read.
    Io(String),
    /// The fixture's files are not what the corpus promises.
    Fixture(String),
    /// PVec's reader refused the artifact.
    Read(String),
}

impl fmt::Display for DriftError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(s) => write!(f, "io: {s}"),
            Self::Fixture(s) => write!(f, "fixture: {s}"),
            Self::Read(s) => write!(f, "the artifact does not read: {s}"),
        }
    }
}

impl std::error::Error for DriftError {}

// ------------------------------------------------------------------------------- entry points

/// Stub: reads the fixture and reports no drift.
pub fn check_fixture(
    _fixture_dir: &Path,
    _pvec: &Path,
    _accepted: Option<&str>,
) -> Result<DriftReport, DriftError> {
    Err(DriftError::Fixture("not implemented".into()))
}

/// Stub: reports no drift.
pub fn check(
    fixture: &str,
    evidence_class: &str,
    _truth: Option<&Value>,
    _accepted: Option<&str>,
    _pvec: &[u8],
) -> Result<DriftReport, DriftError> {
    Ok(DriftReport {
        fixture: fixture.to_owned(),
        evidence_class: evidence_class.to_owned(),
        gating: evidence_class != "QUARANTINED",
        drifts: Vec::new(),
        digest: String::new(),
    })
}

/// Stub: an empty dump.
pub fn canonical_dump(_fixture: &str, _pvec: &[u8]) -> Result<String, DriftError> {
    Ok(String::new())
}

/// Stub: an empty digest.
pub fn digest(_dump: &str) -> String {
    String::new()
}

/// Stub: no section.
pub fn accepted_section<'a>(_file: &'a str, _fixture: &str) -> Option<&'a str> {
    None
}
