//! The conversion report (issue #1093, N6.7): what became of every entity a conversion met.
//!
//! EPIC N's rule is that every conversion emits a machine-readable fidelity and unsupported
//! report. The failure it prevents is the quiet one: a drawing converts, looks plausible, and
//! nothing says that 400 entities were dropped as unsupported or filtered by policy. So the
//! report is built through [`ReportBuilder`], and [`ReportBuilder::finish`] refuses to produce one
//! whose rows do not add up to the number of entities the conversion saw arrive, counted
//! independently of the rows.
//!
//! # Shape
//!
//! One [`EntityRow`] per entity, in id order, then the aggregates: counts by fidelity class,
//! source class and semantic type; the view decisions; the xrefs; the bounds method split. The
//! JSON is pretty-printed from structs and `BTreeMap`s only, so two runs on the same input give
//! the same bytes, and the order rows were pushed in (one worker or eight) does not show.
//! `docs/pvec-conversion-report.md` documents every field and the rule for adding one.
//!
//! # Versioning
//!
//! [`SCHEMA_VERSION`] is in the JSON as `schema_version`. A reader refuses a version it does not
//! know. Adding a field is a version bump unless it is optional and ignorable, and
//! `tests/pvec_1093_report.rs` pins the full field list so a change cannot slip in unnoticed.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::fidelity::{Classification, FidelityClass};
use super::ordering::EntityId;
use super::view_profile::ViewReport;
use super::xref::XrefReport;

/// The version of the JSON schema this module writes and reads.
pub const SCHEMA_VERSION: u32 = 1;

/// One entity: where it came from, what it became, and why when it did not become geometry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntityRow {
    /// The file-local id from the ordering pass.
    pub id: u64,
    /// The drawing's class name (`LINE`, `SPLINE`).
    pub source_class: String,
    /// The source handle, when the drawing gave one.
    pub source_handle: Option<u64>,
    /// The PVec semantic type, `None` for an entity that produced no geometry.
    pub semantic_type: Option<String>,
    /// The fidelity class, spelled as [`FidelityClass::tag`].
    pub fidelity: String,
    /// The reason code for a refusing class, spelled as `ReasonCode::code`.
    pub reason: Option<String>,
    /// The view-profile field that excluded the entity, for `IgnoredByPolicy`.
    pub policy_field: Option<String>,
    /// Warnings recorded against the entity.
    pub warnings: Vec<String>,
}

impl EntityRow {
    /// The row for one classified entity.
    #[must_use]
    pub fn new(
        id: EntityId,
        classification: &Classification,
        source_handle: Option<u64>,
        semantic_type: Option<&str>,
        warnings: Vec<String>,
    ) -> Self {
        EntityRow {
            id: id.get(),
            source_class: classification.source_class().to_owned(),
            source_handle,
            semantic_type: semantic_type.map(str::to_owned),
            fidelity: classification.class().tag().to_owned(),
            reason: classification.reason().map(|r| r.code().to_owned()),
            policy_field: classification.policy_field().map(str::to_owned),
            warnings,
        }
    }
}

/// Counts over all the rows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Totals {
    /// Entities the conversion met. Equal to the rows, and to the sum of `by_fidelity`.
    pub encountered: u64,
    /// Rows per fidelity class. All six classes are present, with zeros.
    pub by_fidelity: BTreeMap<String, u64>,
    /// Rows per source class.
    pub by_source_class: BTreeMap<String, u64>,
    /// Rows per semantic type, for the entities that have one.
    pub by_semantic_type: BTreeMap<String, u64>,
}

/// The view decisions of the conversion (N4.6).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ViewSummary {
    /// The space that was drawn.
    pub space: String,
    /// Leaves kept.
    pub included: u64,
    /// Leaves excluded, by the profile field responsible.
    pub excluded_by_field: BTreeMap<String, u64>,
    /// Leaves excluded, by layer.
    pub excluded_by_layer: BTreeMap<String, u64>,
}

impl ViewSummary {
    /// The summary of a view report.
    #[must_use]
    pub fn from_report(report: &ViewReport) -> Self {
        ViewSummary {
            space: report.space_name.clone(),
            included: report.included,
            excluded_by_field: report
                .excluded_by_field
                .iter()
                .map(|(k, v)| ((*k).to_owned(), *v))
                .collect(),
            excluded_by_layer: report.excluded_by_layer.clone(),
        }
    }
}

/// One xref that was not resolved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnresolvedXrefRow {
    /// The xref's name.
    pub name: String,
    /// The path as the drawing attached it.
    pub path: String,
    /// The path under the root, when the attachment was acceptable.
    pub resolved_path: Option<String>,
    /// Why the path was refused, when it was.
    pub path_refusal: Option<String>,
}

/// The xrefs of the conversion (N4.5).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct XrefSummary {
    /// Xrefs that were left out, in the order they were met.
    pub unresolved: Vec<UnresolvedXrefRow>,
    /// Names of the xrefs that were resolved, sorted.
    pub resolved: Vec<String>,
}

impl XrefSummary {
    /// The summary of an xref report.
    #[must_use]
    pub fn from_report(report: &XrefReport) -> Self {
        let mut resolved = report.resolved.clone();
        resolved.sort();
        XrefSummary {
            unresolved: report
                .unresolved
                .iter()
                .map(|x| UnresolvedXrefRow {
                    name: x.name.clone(),
                    path: x.path.clone(),
                    resolved_path: x
                        .resolved_path
                        .as_ref()
                        .map(|p| p.to_string_lossy().into_owned()),
                    path_refusal: x.path_refusal.map(|r| format!("{r:?}")),
                })
                .collect(),
            resolved,
        }
    }
}

/// How the entity boxes were obtained (N6.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct BoundsSplit {
    /// Boxes in closed form.
    pub exact: u64,
    /// Boxes that are the control hull, enclosing and not tight.
    pub hull_upper_bound: u64,
}

/// The capability profile a conversion ran under (N7.2). Optional and ignorable: a reader that
/// does not know it loses nothing it needs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapabilitySummary {
    /// The named profile, or `Custom` when the caller set the capabilities or the policy.
    pub profile: String,
    /// What happens to an entity outside the set: `RejectInput`, `RejectEntity` or `Downgrade`.
    pub policy: String,
    /// Whether only `SupportedExact` entities passed.
    pub exact_only: bool,
    /// The allowed PVec capability ids, ascending.
    pub allowed: Vec<u32>,
}

/// The whole report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConversionReport {
    /// [`SCHEMA_VERSION`] at the time of writing.
    pub schema_version: u32,
    /// Aggregates.
    pub totals: Totals,
    /// The view decisions.
    pub view: ViewSummary,
    /// The xrefs.
    pub xrefs: XrefSummary,
    /// The bounds method split.
    pub bounds: BoundsSplit,
    /// The capability profile the conversion ran under. Absent when none was recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub capabilities: Option<CapabilitySummary>,
    /// One row per entity, in id order.
    pub entities: Vec<EntityRow>,
}

/// Why a report was not produced, or not read.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ReportError {
    /// The rows are not the entities the conversion met: one was dropped or counted twice.
    #[error("{rows} rows for {encountered} entities encountered")]
    CountMismatch {
        /// Rows pushed.
        rows: u64,
        /// Entities the conversion met.
        encountered: u64,
    },
    /// Two rows carry one id.
    #[error("entity {0} has two rows")]
    DuplicateId(u64),
    /// A row's class, reason and policy field disagree: a refusing class without its reason, an
    /// `IgnoredByPolicy` without its field, or a reason on a class that has none.
    #[error("entity {0} has a fidelity class that disagrees with its reason or policy field")]
    Inconsistent(u64),
    /// A row names a fidelity class this schema version does not have.
    #[error("entity {id} names the unknown fidelity class {class:?}")]
    UnknownClass {
        /// The entity.
        id: u64,
        /// The spelling.
        class: String,
    },
    /// The JSON is not a report of a version this reader knows.
    #[error("the JSON is not a version {SCHEMA_VERSION} conversion report")]
    Schema(#[source] Option<serde_json::Error>),
    /// Serialising failed.
    #[error("serialising the report failed")]
    Json(#[source] serde_json::Error),
}

/// Collects rows. The order they are pushed in does not matter.
#[derive(Debug, Default)]
pub struct ReportBuilder {
    rows: Vec<EntityRow>,
}

impl ReportBuilder {
    /// An empty builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a row.
    pub fn push(&mut self, row: EntityRow) {
        self.rows.push(row);
    }

    /// Closes the report against the number of entities the conversion saw arrive, counted
    /// independently of the rows.
    ///
    /// # Errors
    ///
    /// [`ReportError::CountMismatch`] when the rows are not exactly the entities encountered,
    /// [`ReportError::DuplicateId`], [`ReportError::UnknownClass`] and
    /// [`ReportError::Inconsistent`] for rows that cannot be right.
    pub fn finish(
        mut self,
        encountered: u64,
        view: ViewSummary,
        xrefs: XrefSummary,
        bounds: BoundsSplit,
    ) -> Result<ConversionReport, ReportError> {
        let rows = self.rows.len() as u64;
        if rows != encountered {
            return Err(ReportError::CountMismatch { rows, encountered });
        }
        self.rows.sort_by_key(|r| r.id);
        let mut seen = BTreeSet::new();
        let mut totals = Totals {
            encountered,
            by_fidelity: FidelityClass::ALL
                .iter()
                .map(|c| (c.tag().to_owned(), 0))
                .collect(),
            by_source_class: BTreeMap::new(),
            by_semantic_type: BTreeMap::new(),
        };
        for r in &self.rows {
            if !seen.insert(r.id) {
                return Err(ReportError::DuplicateId(r.id));
            }
            let class = FidelityClass::ALL
                .into_iter()
                .find(|c| c.tag() == r.fidelity)
                .ok_or_else(|| ReportError::UnknownClass {
                    id: r.id,
                    class: r.fidelity.clone(),
                })?;
            let refusing = matches!(
                class,
                FidelityClass::RequiresPVecCapability
                    | FidelityClass::RequiresExtension
                    | FidelityClass::Unsupported
            );
            let consistent = r.reason.is_some() == refusing
                && r.policy_field.is_some() == (class == FidelityClass::IgnoredByPolicy)
                && match class {
                    FidelityClass::SupportedExact | FidelityClass::SupportedEquivalent => {
                        r.semantic_type.is_some()
                    }
                    // The geometry may exist in part, so a type is allowed but not required.
                    FidelityClass::RequiresExtension => true,
                    _ => r.semantic_type.is_none(),
                };
            if !consistent {
                return Err(ReportError::Inconsistent(r.id));
            }
            *totals.by_fidelity.entry(r.fidelity.clone()).or_default() += 1;
            *totals
                .by_source_class
                .entry(r.source_class.clone())
                .or_default() += 1;
            if let Some(t) = &r.semantic_type {
                *totals.by_semantic_type.entry(t.clone()).or_default() += 1;
            }
        }
        Ok(ConversionReport {
            schema_version: SCHEMA_VERSION,
            totals,
            view,
            xrefs,
            bounds,
            capabilities: None,
            entities: self.rows,
        })
    }
}

impl ConversionReport {
    /// The report as pretty JSON. The same report gives the same bytes.
    ///
    /// # Errors
    ///
    /// [`ReportError::Json`] if serialising fails.
    pub fn to_json_string(&self) -> Result<String, ReportError> {
        serde_json::to_string_pretty(self).map_err(ReportError::Json)
    }

    /// Reads a report, refusing a `schema_version` other than [`SCHEMA_VERSION`].
    ///
    /// # Errors
    ///
    /// [`ReportError::Schema`] for malformed JSON or another version.
    pub fn from_json_slice(bytes: &[u8]) -> Result<Self, ReportError> {
        let report: ConversionReport =
            serde_json::from_slice(bytes).map_err(|e| ReportError::Schema(Some(e)))?;
        if report.schema_version == SCHEMA_VERSION {
            Ok(report)
        } else {
            Err(ReportError::Schema(None))
        }
    }
}
