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
    /// The viewport a model-space entity was drawn through on a sheet (#1563), by handle. Absent
    /// for everything else.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub viewport: Option<u64>,
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
            viewport: None,
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

/// What a PDF page conversion did beside its vector entities: the images, the fills and the text
/// (issue #1491). The page's coordinates are in points. Every map is keyed by the reason or note key a `docs/pvec-*.md` page names.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PdfSummary {
    /// The 1-based page converted.
    pub page: u64,
    /// Images on the page.
    pub images: u64,
    /// Images written as raster blocks.
    pub rasters: u64,
    /// Images refused, by reason key.
    pub images_refused: BTreeMap<String, u64>,
    /// Written images that carry a note, by note key.
    pub image_notes: BTreeMap<String, u64>,
    /// Fills, shadings, patterns and images the fill policy counted, by reason key.
    pub fills: BTreeMap<String, u64>,
    /// Text items omitted.
    pub text_omitted: u64,
    /// What the page said about its scale and place (#1547). Optional and ignorable, like
    /// [`ComponentsSummary`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration: Option<PdfCalibrationSummary>,
}

/// What a PDF conversion read about the page's scale and place (#1547), and what it made of it.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PdfCalibrationSummary {
    /// The `/VP` viewports on the page.
    pub viewports: u64,
    /// The `/LGIDict` georeferences on the page (#1548).
    #[serde(default)]
    pub lgidicts: u64,
    /// The index in `/VP` of the viewport chosen for its measure (even one that could not be used,
    /// which `notes` then says), absent when no viewport has a measure.
    pub viewport: Option<u64>,
    /// The scale's provenance: `declared-unit` (the point) or `plan-scale` (the viewport's `/RL`).
    pub scale: String,
    /// The georeference's CRS id, or `definition` when it is named by its definition only. Absent
    /// when there is no georeference.
    pub georeference: Option<String>,
    /// Why something the page said was not used, by key (`rl-unit-unknown`,
    /// `geo-projected-without-definition`, `lgidict-not-mapped`), in the order met.
    pub notes: Vec<String>,
}

/// What the component streams of a conversion did (#1525). Optional and ignorable: a reader that
/// does not know it loses nothing it needs.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct ComponentsSummary {
    /// The streams written, by name (`layer`, `line-style`, `provenance`), in file order.
    pub streams: Vec<String>,
    /// Rows left out because PVec cannot hold them, keyed `stream.reason` (`layer.name-too-long`).
    pub dropped: BTreeMap<String, u64>,
    /// DWG colour indices seen and dropped under the monochrome policy. Counted only when the line style
    /// stream is selected, because that is the pass that reads an entity's style.
    pub colours_ignored: u64,
}

/// What the room and corridor step of a conversion found (#1526). Optional and ignorable, like
/// [`ComponentsSummary`]: the regions are a guess from the wall linework, and the decoded geometry and
/// the manifest box are the same whether or not the step ran.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct RegionsSummary {
    /// Rooms found.
    pub rooms: u64,
    /// Corridors found.
    pub corridors: u64,
    /// Gaps between wall ends the step closed. The closing segments are never written as geometry.
    pub gaps_closed: u64,
    /// Faces no rule could label.
    pub unlabelled: u64,
    /// Regions that found a text label inside them.
    pub labelled: u64,
    /// Regions written to PVec's semantic class stream.
    pub written: u64,
    /// Regions left out of that stream, keyed by reason (`derived-boundary`, `unlabelled`,
    /// `unwritten-boundary`, `no-boundary`, `ring-too-long`).
    pub dropped: BTreeMap<String, u64>,
    /// The reason code when the step refused (`regions.node-limit`, `regions.edge-limit`,
    /// `regions.face-limit`). The conversion carries on without regions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refused: Option<String>,
}

/// What the print profile made of a paper layout (#1562): the sheet's frame, its unit, the plot
/// settings as the drawing stores them, and what the sheet clip did. Optional and ignorable, like
/// [`ComponentsSummary`]; absent for model space. Numbers the drawing stores as `f64` are written as
/// their shortest round-trip decimal text, so the report keeps `Eq`.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SheetSummary {
    /// The layout's name.
    pub layout: String,
    /// `paper` when the frame is the paper sheet, `extents` when it is the kept geometry's extent.
    pub frame: String,
    /// Why the frame is the extents: `plot-settings-absent`, `plot-settings-not-reportable` or
    /// `paper-size-zero`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub frame_reason: Option<String>,
    /// Where the paper frame's position came from: `unmeasured`, because the formula from the plot
    /// origin and the margins has no measured fixture yet (the print profile ADR's open item).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<String>,
    /// The sheet's unit, as `docs/pvec-units.md` names it.
    pub unit: String,
    /// Where the unit came from: `plot-paper-units`, or `drawing` (the unit the conversion already
    /// had).
    pub unit_source: String,
    /// The paper sheet, `[min x, min y, max x, max y]` in the sheet unit, with the paper frame.
    #[serde(rename = "box", default, skip_serializing_if = "Option::is_none")]
    pub sheet_box: Option<[String; 4]>,
    /// The plot settings as the drawing stores them, absent when it stores none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub plot: Option<SheetPlotSummary>,
    /// The plot scale as written into the calibration record, `numerator:denominator`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<String>,
    /// The artifact's calibration: `plot-scale`, `plot-scale-uncertain`, `caller` or `none`.
    pub calibration: String,
    /// Stroke widths were left out of the line-style stream because the layout does not print
    /// lineweights.
    pub widths_dropped: bool,
    /// One row per VIEWPORT entity on the sheet, in the drawing's order: what it did and why
    /// (#1563).
    pub viewports: Vec<SheetViewport>,
    /// What the sheet clip did, by key: `outside` (dropped, wholly outside), `crossing` (a curve
    /// that crosses the edge, refused), `cut` (a line or polyline cut at the edge) and `pieces`
    /// (what the cut ones became).
    pub clip: BTreeMap<String, u64>,
    /// What the drawing said that was reported and not applied, by key, in a fixed order.
    pub notes: Vec<String>,
}

/// One viewport of a sheet in the report's [`SheetSummary`] (#1563).
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SheetViewport {
    /// The VIEWPORT entity's handle.
    pub handle: u64,
    /// The viewport's number as the reader gives it; `0` when the provider reports no viewport for
    /// the entity.
    pub id: i16,
    /// `drawn` (model space shows through it), `paper-view` (the sheet's own view, not a window
    /// onto the model), `off`, `hide-plot`, `no-plot-layer`, or `refused`.
    pub status: String,
    /// Why a refused viewport is refused: `non-plan`, `perspective`, `view-height`, `twist`,
    /// `clip-polygon`, `clip-circle`, `clip-ellipse`, `clip-spline`, `clip-other` or
    /// `not-reported`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// For a drawn viewport, its scale as `1:N`: one paper unit shows `N` model units.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scale: Option<String>,
    /// For a drawn viewport, its rectangle on the sheet, `[min x, min y, max x, max y]`.
    #[serde(rename = "box", default, skip_serializing_if = "Option::is_none")]
    pub paper_box: Option<[String; 4]>,
    /// For a drawn viewport, the rows written through it.
    pub entities: u64,
}

/// A layout's plot settings in the report's [`SheetSummary`], as the drawing stores them.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SheetPlotSummary {
    /// The paper size's name.
    pub paper_size: String,
    /// The paper's width in millimetres.
    pub paper_width: String,
    /// The paper's height in millimetres.
    pub paper_height: String,
    /// The plot paper units code.
    pub paper_units: i16,
    /// The plot rotation code, reported and not applied.
    pub rotation: i16,
    /// The plot type code, reported and not applied.
    pub plot_type: i16,
    /// The plot scale's paper side, as stored.
    pub scale_numerator: String,
    /// The plot scale's drawing side, as stored.
    pub scale_denominator: String,
    /// The plot style table the layout names; the file is never read.
    pub style_sheet: String,
    /// Lineweights print.
    pub print_lineweights: bool,
    /// Lineweights scale with the plot scale; reported only.
    pub scale_lineweights: bool,
    /// Plot styles apply; reported only.
    pub plot_plot_styles: bool,
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
    /// The file name of the georeference sidecar a conversion used to publish beside the artifact
    /// (N7.3). No conversion writes one since #1554, so it is always absent now; it stays so older
    /// reports still read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub georeference: Option<String>,
    /// The diagnostics of the conversion, in entity id order (N7.5). Its own section, never mixed
    /// into the geometry; absent when the conversion made none.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub diagnostics: Vec<super::diagnostics::Diagnostic>,
    /// What the component streams did (#1525). Absent when the conversion wrote none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub components: Option<ComponentsSummary>,
    /// What the room and corridor step found (#1526). Absent when the step was not asked for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regions: Option<RegionsSummary>,
    /// The images, fills and text of a PDF page. Absent for a DWG.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pdf: Option<PdfSummary>,
    /// The paper sheet of a print-profile conversion (#1562). Absent for model space.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sheet: Option<SheetSummary>,
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

    /// Rewrites the row of `id` to a refusing class, for an entity the writer refused after it was
    /// classified. The row loses its semantic type and gains `warning`. Nothing happens when no
    /// row has that id.
    pub fn refuse(&mut self, id: EntityId, classification: &Classification, warning: String) {
        if let Some(r) = self.rows.iter_mut().find(|r| r.id == id.get()) {
            r.fidelity = classification.class().tag().to_owned();
            r.reason = classification.reason().map(|c| c.code().to_owned());
            r.semantic_type = None;
            r.warnings.push(warning);
        }
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
            georeference: None,
            diagnostics: Vec::new(),
            components: None,
            regions: None,
            pdf: None,
            sheet: None,
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

    /// Writes the same bytes as [`to_json_string`](Self::to_json_string) to `out`, without building
    /// the string.
    ///
    /// # Errors
    ///
    /// [`ReportError::Json`] when serialisation or `out` fails.
    pub fn write_json(&self, out: &mut dyn std::io::Write) -> Result<(), ReportError> {
        serde_json::to_writer_pretty(out, self).map_err(ReportError::Json)
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
