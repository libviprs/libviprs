//! Structured conversion diagnostics (issue #1416, N7.5): the per-entity audit trail, kept outside
//! the geometry hot path, in machine-readable form.
//!
//! A [`Diagnostic`] says what happened to one entity or to the run: the source, the layout, the
//! entity's handle and class, the block path it was reached through, its layer, the stage of the
//! conversion, a severity, a stable [`DiagnosticCode`] and a stable `message_id`. The wording in
//! `message` is for people and may change; the code, the severity and the message id may not, and
//! the severity is a function of the code ([`DiagnosticCode::severity`]), never parsed from text.
//!
//! The closed table of codes is [`DiagnosticCode::ALL`], documented in `docs/pvec-diagnostics.md`
//! with the fixture that emits each one. `tests/pvec_1416_diagnostics.rs` reads the table from the
//! document and fails on a code nothing emits, and on one that is emitted but not written down.
//!
//! # Where they go
//!
//! An observer attached to the conversion receives each one as it is made
//! (`EngineEvent::ConversionDiagnostic`), so diagnostics stream out. The durable copy is the
//! `diagnostics` section of the conversion report, its own stream and never mixed into geometry.
//! A supported entity makes none, so with no observer attached the per-entity path of a drawing
//! that converts cleanly allocates nothing for diagnostics.

use serde::{Deserialize, Serialize};

/// How serious a diagnostic is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Severity {
    /// Expected and harmless: an entity the view profile hid, a run the caller cancelled.
    Info,
    /// An entity or a block did not become geometry.
    Warning,
    /// The run stopped.
    Error,
}

/// Where in the conversion a diagnostic arose.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[non_exhaustive]
pub enum Stage {
    /// Opening the source.
    Open,
    /// Applying the view profile.
    View,
    /// Expanding blocks.
    Flatten,
    /// Classifying and mapping entities.
    Classify,
    /// Writing the artifact.
    Write,
    /// Building the report and the sidecar.
    Report,
    /// Publishing the files.
    Publish,
}

/// The closed set of diagnostic codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DiagnosticCode {
    /// An entity needs a PVec capability that is not allowed or not available.
    EntityRequiresCapability,
    /// An entity has no mapping into PVec, or PVec refused its geometry.
    EntityUnsupported,
    /// The view profile left an entity out.
    EntityIgnoredByPolicy,
    /// An INSERT was not expanded.
    InsertNotExpanded,
    /// The source could not be opened.
    SourceOpenFailed,
    /// The view profile could not be applied.
    ViewFailed,
    /// Block expansion failed or a limit was exceeded.
    BlocksFailed,
    /// No unit was given and the provider reports none.
    UnitRequired,
    /// An entity needs a capability the run does not allow, and the policy is to stop.
    CapabilityRejected,
    /// PVec's writer refused the artifact.
    WriteFailed,
    /// The report could not be built, or its counts disagree with the entities met.
    ReportFailed,
    /// The artifact and report could not be published.
    PublishFailed,
    /// The georeference sidecar could not be built or written.
    GeoreferenceFailed,
    /// The caller cancelled the run.
    Cancelled,
}

impl DiagnosticCode {
    /// Every code, in the order `docs/pvec-diagnostics.md` lists them.
    pub const ALL: [DiagnosticCode; 14] = [
        DiagnosticCode::EntityRequiresCapability,
        DiagnosticCode::EntityUnsupported,
        DiagnosticCode::EntityIgnoredByPolicy,
        DiagnosticCode::InsertNotExpanded,
        DiagnosticCode::SourceOpenFailed,
        DiagnosticCode::ViewFailed,
        DiagnosticCode::BlocksFailed,
        DiagnosticCode::UnitRequired,
        DiagnosticCode::CapabilityRejected,
        DiagnosticCode::WriteFailed,
        DiagnosticCode::ReportFailed,
        DiagnosticCode::PublishFailed,
        DiagnosticCode::GeoreferenceFailed,
        DiagnosticCode::Cancelled,
    ];

    /// The code as the document and the JSON spell it.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            DiagnosticCode::EntityRequiresCapability => "ENTITY_REQUIRES_CAPABILITY",
            DiagnosticCode::EntityUnsupported => "ENTITY_UNSUPPORTED",
            DiagnosticCode::EntityIgnoredByPolicy => "ENTITY_IGNORED_BY_POLICY",
            DiagnosticCode::InsertNotExpanded => "INSERT_NOT_EXPANDED",
            DiagnosticCode::SourceOpenFailed => "SOURCE_OPEN_FAILED",
            DiagnosticCode::ViewFailed => "VIEW_FAILED",
            DiagnosticCode::BlocksFailed => "BLOCKS_FAILED",
            DiagnosticCode::UnitRequired => "UNIT_REQUIRED",
            DiagnosticCode::CapabilityRejected => "CAPABILITY_REJECTED",
            DiagnosticCode::WriteFailed => "WRITE_FAILED",
            DiagnosticCode::ReportFailed => "REPORT_FAILED",
            DiagnosticCode::PublishFailed => "PUBLISH_FAILED",
            DiagnosticCode::GeoreferenceFailed => "GEOREFERENCE_FAILED",
            DiagnosticCode::Cancelled => "CANCELLED",
        }
    }

    /// The severity of the code. A property of the code, so rewording a message cannot move it.
    #[must_use]
    pub const fn severity(self) -> Severity {
        match self {
            DiagnosticCode::EntityIgnoredByPolicy | DiagnosticCode::Cancelled => Severity::Info,
            DiagnosticCode::EntityRequiresCapability
            | DiagnosticCode::EntityUnsupported
            | DiagnosticCode::InsertNotExpanded => Severity::Warning,
            DiagnosticCode::SourceOpenFailed
            | DiagnosticCode::ViewFailed
            | DiagnosticCode::BlocksFailed
            | DiagnosticCode::UnitRequired
            | DiagnosticCode::CapabilityRejected
            | DiagnosticCode::WriteFailed
            | DiagnosticCode::ReportFailed
            | DiagnosticCode::PublishFailed
            | DiagnosticCode::GeoreferenceFailed => Severity::Error,
        }
    }

    /// The stage the code belongs to.
    #[must_use]
    pub const fn stage(self) -> Stage {
        match self {
            DiagnosticCode::EntityRequiresCapability
            | DiagnosticCode::EntityUnsupported
            | DiagnosticCode::EntityIgnoredByPolicy
            | DiagnosticCode::InsertNotExpanded
            | DiagnosticCode::CapabilityRejected
            | DiagnosticCode::Cancelled => Stage::Classify,
            DiagnosticCode::SourceOpenFailed | DiagnosticCode::UnitRequired => Stage::Open,
            DiagnosticCode::ViewFailed => Stage::View,
            DiagnosticCode::BlocksFailed => Stage::Flatten,
            DiagnosticCode::WriteFailed => Stage::Write,
            DiagnosticCode::ReportFailed | DiagnosticCode::GeoreferenceFailed => Stage::Report,
            DiagnosticCode::PublishFailed => Stage::Publish,
        }
    }
}

/// One diagnostic.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Diagnostic {
    /// The stable code.
    pub code: DiagnosticCode,
    /// [`DiagnosticCode::severity`] of `code`.
    pub severity: Severity,
    /// [`DiagnosticCode::stage`] of `code`.
    pub stage: Stage,
    /// A stable identifier of the message: the PVec reason code for an entity, the failing step
    /// for an error. Wording changes do not move it.
    pub message_id: String,
    /// The message for people. Not stable.
    pub message: String,
    /// The source file, when the conversion has a path for it.
    pub source: Option<String>,
    /// The layout the entity belongs to.
    pub layout: Option<String>,
    /// The entity's handle, when the drawing gave one.
    pub handle: Option<u64>,
    /// The drawing's class name for the entity.
    pub class: Option<String>,
    /// The INSERT chain the entity was reached through, spelled `block@HANDLE/...`.
    pub block_path: Option<String>,
    /// The entity's own layer.
    pub layer: Option<String>,
}

impl Diagnostic {
    /// A diagnostic of `code` with no entity attached: severity and stage come from the code.
    #[must_use]
    pub fn new(
        code: DiagnosticCode,
        message_id: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Diagnostic {
            code,
            severity: code.severity(),
            stage: code.stage(),
            message_id: message_id.into(),
            message: message.into(),
            source: None,
            layout: None,
            handle: None,
            class: None,
            block_path: None,
            layer: None,
        }
    }
}
