//! The conversion API (issue #1048, N7.1): a DWG, a view profile and resource limits in, a
//! published `.pvec` artifact and its conversion report out.
//!
//! [`PvecConversion`] is the one surface an application calls. It is built the way `EngineBuilder`
//! is, and [`convert`](PvecConversion::convert) drives the pieces the earlier phases built and
//! kept internal:
//!
//! ```text
//! provider (acadrust by default)
//!   -> view profile (layers, depth)     which leaves are drawn, and which were excluded
//!   -> block flattening                 one composed map per leaf
//!   -> extraction + transform promotion one PVec curve per supported leaf, a class for the rest
//!   -> ordering                         dense file-local ids
//!   -> writer funnel                    the artifact
//!   -> conversion report                one row per entity met, checked against the count met
//!   -> atomic publish                   artifact and report, both or neither
//! ```
//!
//! Every entity the conversion meets is counted exactly once: a drawn leaf, a leaf the profile
//! excluded, and an INSERT that could not be expanded. Nothing is tessellated and nothing is
//! repaired; an entity PVec cannot carry is reported with its reason code and left out.
//!
//! The drawing's unit comes from the provider (acadrust reads the header) or from
//! [`PvecConversion::unit`]. A unit PVec cannot declare is refused by the writer, not rescaled.
//!
//! # Example
//!
//! ```
//! use std::path::PathBuf;
//! use libviprs::cad::CadSource;
//! use libviprs::pvec::SourceUnit;
//! use libviprs::pvec::publish::open_published;
//! use libviprs::pvec_convert::PvecConversion;
//!
//! let src = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
//!     .join("tests/fixtures/pvec/provider/vc_layers.dwg");
//! let dir = std::env::temp_dir().join(format!("libviprs-doc-convert-{}", std::process::id()));
//! std::fs::create_dir_all(&dir)?;
//! let dest = dir.join("layers.pvec");
//!
//! // The fixture's header declares no unit, and a unit PVec cannot declare is refused, so say one.
//! let out = PvecConversion::new(CadSource::Path(&src), &dest)
//!     .unit(SourceUnit::Meters)
//!     .convert()?;
//! assert_eq!(out.report.totals.encountered, 9);
//! assert_eq!(open_published(&dest)?.entity_count()?, out.written);
//! # std::fs::remove_dir_all(&dir)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Needs the `pvec-convert` feature.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use thiserror::Error;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

mod calibration;
mod capability;
mod georef;
#[cfg(feature = "pvec-raster")]
mod pdf;
pub use crate::pvec::writer::recalibrate;
pub use calibration::{
    Calibration, CalibrationError, ControlPoint, CrsDefinition, CrsKind, DefinitionFormat,
    Georeference, GeoreferenceMethod, ScaleClaim, ScaleProvenance,
};
pub use capability::{CapabilityPolicy, ConversionProfile};
#[allow(deprecated)]
pub use georef::{
    GeoreferenceError, GeoreferenceHandoff, HandoffV1, SidecarImport, SourceBox, UnitRecord,
    georef_path, import_sidecar,
};
#[cfg(feature = "pvec-raster")]
#[cfg_attr(docsrs, doc(cfg(feature = "pvec-raster")))]
pub use pdf::{PdfConversion, PdfConversionOutput, PdfConvertError};

use crate::CancelToken;
use crate::cad::CadSource;
use crate::observe::{EngineEvent, EngineObserver};
use crate::pvec::bounds::BoundsReport;
use crate::pvec::classify::{ElementClass, ElementRules};
use crate::pvec::components::{ComponentSource, ComponentsError, StreamSelection, build_streams};
use crate::pvec::diagnostics::{Diagnostic, DiagnosticCode};
use crate::pvec::dwg_acadrust::AcadrustProvider;
use crate::pvec::dwg_basic::{self, BasicError};
use crate::pvec::dwg_spline::{self, SplineError};
use crate::pvec::fidelity::{Basis, Classification, FidelityReport, ReasonCode, Tally, TallyError};
use crate::pvec::flatten::{
    BlockResolver, FlatEntity, FlattenBounds, FlattenError, LineageStore, SkipReason, SkippedInsert,
};
use crate::pvec::mapping::{kind_of, row_for};
use crate::pvec::ordering::key_cmp;
use crate::pvec::provenance::NoProvenance;
use crate::pvec::publish::{PublishError, StreamedPublishError, publish_streamed_with};
use crate::pvec::regions::{RegionConfig, RoomLabel, WallEdge, derive_regions, semantic_stream};
use crate::pvec::report::{
    BoundsSplit, CapabilitySummary, ComponentsSummary, ConversionReport, EntityRow, RegionsSummary,
    ReportBuilder, ReportError, SheetPlotSummary, SheetSummary, UnresolvedXrefRow, ViewSummary,
    XrefSummary,
};
use crate::pvec::style::{LineStyleComponent, dwg_line_style, dwg_style_input};
use crate::pvec::transform::{ComposedTransform, promote};
use crate::pvec::writer::{
    CapabilityId, LeafConfig, OpenArtifact, SpillConfig, WriteConfig, WriteError, metres_per_unit,
    open_geometry,
};
use crate::pvec::{
    DwgProvider, EntityId, EntityScope, EntityStyle, ExcludedEntity, FidelityClass, LayoutKind,
    PolicyReason, ProvenanceStub, ProviderError, RawEntity, RawGeometry, SourceFact, SourceLimits,
    SourceUnit, ViewProfile, ViewProfileError, select_layout,
};
use crate::pvec::{Normalized2DGeometry, ViewedEntity};
use crate::sync_queue::bounded;
use pvec_geometry::{Curve, Curve2d, Line, Point2, Trimmed};

/// Entities per batch: the cancel token is looked at, and a progress event is sent, once per batch.
const BATCH: usize = 256;

/// The queue between the classifier and its consumer holds this many outcomes unless
/// [`PvecConversion::queue_capacity`] says otherwise.
const DEFAULT_QUEUE_CAPACITY: usize = 64;

/// Why a conversion failed. Every variant that wraps a lower error keeps it as `source()`.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PvecConvertError {
    /// The provider could not open the source.
    #[error("the source could not be opened")]
    Open(#[source] ProviderError),
    /// The view profile could not be applied.
    #[error("the view profile could not be applied")]
    View(#[source] ViewProfileError),
    /// Block expansion failed.
    #[error("blocks could not be expanded")]
    Flatten(#[source] FlattenError),
    /// No unit was given and the provider cannot read one.
    #[error("the drawing's unit is not known and none was given")]
    UnitRequired,
    /// The writer refused.
    #[error("the artifact could not be written")]
    Write(#[source] WriteError),
    /// PVec's writer refused every curve, so there is no artifact. The report and the counts say
    /// what each entity was, with the refused ones counted as `pvec-cannot-store`; nothing was
    /// published.
    #[error("PVec's writer refused every curve, so no artifact was written")]
    NothingStorable {
        /// The conversion report, as it would have been published beside the artifact.
        report: Box<ConversionReport>,
        /// Entities per fidelity class, counted independently of the report rows.
        fidelity: Box<FidelityReport>,
    },
    /// The class counts disagree with the entities met.
    #[error("the class counts do not add up")]
    Tally(#[source] TallyError),
    /// A component stream could not be built.
    #[error("a component stream could not be built")]
    Components(#[source] ComponentsError),
    /// The report could not be built.
    #[error("the report could not be built")]
    Report(#[source] ReportError),
    /// The artifact and report could not be published.
    #[error("the artifact could not be published")]
    Publish(#[source] PublishError),
    /// An entity needs a PVec capability the run did not allow, and the policy is
    /// [`CapabilityPolicy::RejectInput`].
    #[error(
        "{source_class} (handle {handle:?}) needs the PVec capability {name} ({}), which is not allowed",
        capability.get()
    )]
    CapabilityNotAllowed {
        /// PVec's identifier of the capability.
        capability: CapabilityId,
        /// Its registry name.
        name: &'static str,
        /// The drawing's class name of the entity.
        source_class: String,
        /// The entity's handle, when the drawing gave one.
        handle: Option<u64>,
        /// The entity's own layer.
        layer: String,
        /// The INSERT chain the entity was reached through, spelled `block@HANDLE/...`.
        block_path: String,
    },
    /// The cancel token fired. Nothing was published.
    #[error("cancelled")]
    Cancelled,
}

/// What a conversion produced.
#[derive(Debug)]
pub struct PvecConversionOutput {
    /// The published artifact, readable with [`open_published`](crate::pvec::publish::open_published).
    pub artifact: PathBuf,
    /// The conversion report, also published beside the artifact.
    pub report: ConversionReport,
    /// Entities per fidelity class, counted independently of the report rows.
    pub fidelity: FidelityReport,
    /// Entities written into the artifact.
    pub written: u64,
    /// The most outcomes that were waiting for the consumer at once. Never above the capacity set
    /// with [`PvecConversion::queue_capacity`].
    pub peak_queue_depth: u32,
}

/// One conversion, configured the way `EngineBuilder` configures a pyramid run.
pub struct PvecConversion<'a> {
    source: CadSource<'a>,
    dest: &'a Path,
    view_profile: ViewProfile,
    unit: Option<SourceUnit>,
    leaves: Option<LeafConfig>,
    spill: Option<SpillConfig>,
    cancel: CancelToken,
    gate: Gate,
    calibration: Option<Calibration>,
    observer: Option<Arc<dyn EngineObserver>>,
    queue_capacity: usize,
    components: StreamSelection,
    regions: Option<RegionConfig>,
}

impl<'a> PvecConversion<'a> {
    /// A conversion of `source` into `dest`, under the default view profile.
    #[must_use]
    pub fn new(source: CadSource<'a>, dest: &'a Path) -> Self {
        Self {
            source,
            dest,
            view_profile: ViewProfile::default_profile(),
            unit: None,
            leaves: None,
            spill: None,
            cancel: CancelToken::new(),
            gate: Gate::of(ConversionProfile::Default),
            calibration: None,
            observer: None,
            queue_capacity: DEFAULT_QUEUE_CAPACITY,
            components: StreamSelection::NONE,
            regions: None,
        }
    }

    /// The component streams to write beside the geometry: layer, line style and provenance, each
    /// on its own (#1525). Off by default, and the decoded geometry and the manifest box are the same either way.
    #[must_use]
    pub fn components(mut self, select: StreamSelection) -> Self {
        self.components = select;
        self
    }

    /// Derive rooms and corridors from the wall linework and write them as PVec's semantic class
    /// stream (#1526). Off (`None`) by default. The walls are the straight lines the default
    /// [`ElementRules`] call a wall, the room names come from TEXT, and the decoded
    /// geometry and the manifest box are the same either way: the segments that close a door gap are
    /// never written as geometry, so a room that runs along one is left out of the stream and counted
    /// in the report. Explicit rooms are not passed, so only inferred regions come out.
    #[must_use]
    pub fn regions(mut self, config: Option<RegionConfig>) -> Self {
        self.regions = config;
        self
    }

    /// The view profile (N3.3).
    #[must_use]
    pub fn view_profile(mut self, profile: ViewProfile) -> Self {
        self.view_profile = profile;
        self
    }

    /// The resource ceilings (N3.6), carried on the view profile.
    #[must_use]
    pub fn limits(mut self, limits: SourceLimits) -> Self {
        self.view_profile = self.view_profile.with_limits(limits);
        self
    }

    /// The drawing's unit, overriding what the provider reads. A paper sheet whose plot settings
    /// name inches or millimetres is in that unit instead (P-6, #1562).
    #[must_use]
    pub fn unit(mut self, unit: SourceUnit) -> Self {
        self.unit = Some(unit);
        self
    }

    /// Lay a spatial directory and Tier 0 blocks into the artifact. Without it the artifact is
    /// a plain file.
    #[must_use]
    pub fn leaves(mut self, leaves: LeafConfig) -> Self {
        self.leaves = Some(leaves);
        self
    }

    /// Stream the artifact through PVec's spill writer: its pushes are kept in files under `spill.root` and its
    /// memory stays inside `spill.budget`, and the artifact is written straight into the `.pvec.partial` file.
    /// The bytes and the report are those of a conversion without it. The memory is bounded when
    /// [`PvecConversion::leaves`] is set too (a plain artifact holds one block per geometry type).
    #[must_use]
    pub fn spill(mut self, spill: SpillConfig) -> Self {
        self.spill = Some(spill);
        self
    }

    /// The cancel token, polled before any file exists and while entities are classified.
    #[must_use]
    pub fn cancel(mut self, cancel: CancelToken) -> Self {
        self.cancel = cancel;
        self
    }

    /// Receives each diagnostic as it is made (`EngineEvent::ConversionDiagnostic`), and the one
    /// the run fails with. Without one, a drawing that converts cleanly makes no diagnostic at all.
    #[must_use]
    pub fn observer(mut self, observer: Arc<dyn EngineObserver>) -> Self {
        self.observer = Some(observer);
        self
    }

    /// How many classified entities may wait for the consumer. The classifier blocks when the
    /// queue is full, so memory stays bounded whatever the consumer's speed; the deepest the queue
    /// was is [`PvecConversionOutput::peak_queue_depth`]. Tuning it is N8.6's measured decision;
    /// the default is a placeholder that is safe, not tuned.
    #[must_use]
    pub fn queue_capacity(mut self, capacity: usize) -> Self {
        self.queue_capacity = capacity.max(1);
        self
    }

    /// The drawing's calibration, written into the artifact's calibration block (issue #1552). A
    /// DWG gets none unless it is given here: its `$INSUNITS` is already the manifest's unit.
    ///
    /// Control points go in the calibration's [`Georeference`], with both their sides. The
    /// source-only `control_points` builder is gone (issue #1556):
    ///
    /// ```compile_fail,E0599
    /// use std::path::Path;
    /// use libviprs::cad::CadSource;
    /// use libviprs::pvec_convert::PvecConversion;
    ///
    /// let _ = PvecConversion::new(CadSource::Bytes(b""), Path::new("o.pvec"))
    ///     .control_points([[0.0, 0.0]]);
    /// ```
    #[must_use]
    pub fn calibration(mut self, calibration: Calibration) -> Self {
        self.calibration = Some(calibration);
        self
    }

    /// A named capability set, exactness rule and policy: [`ConversionProfile::Strict`],
    /// `Default` (the default) or `Permissive`. Replaces any earlier capability setting.
    #[must_use]
    pub fn profile(mut self, profile: ConversionProfile) -> Self {
        self.gate = Gate::of(profile);
        self
    }

    /// The PVec capabilities this run may use, replacing the profile's set. An entity that needs
    /// another is handled by the [`capability_policy`](Self::capability_policy). The profile's
    /// exactness rule is kept, and the report names the profile as `Custom`.
    #[must_use]
    pub fn allowed_capabilities(mut self, allowed: impl IntoIterator<Item = CapabilityId>) -> Self {
        self.gate.allowed = allowed.into_iter().collect();
        self.gate.name = "Custom";
        self
    }

    /// What happens to an entity that needs a capability outside the allowed set.
    #[must_use]
    pub fn capability_policy(mut self, policy: CapabilityPolicy) -> Self {
        self.gate.policy = policy;
        self.gate.name = "Custom";
        self
    }

    /// Classifies the drawing without writing anything and returns the per-class counts a
    /// [`convert`](Self::convert) would report, before PVec's writer has seen anything: an entity
    /// counted as supported here can still come back as `pvec-cannot-store` from the conversion,
    /// because the rule is PVec's and is not copied. Under [`CapabilityPolicy::RejectInput`] it fails
    /// the way the conversion would.
    ///
    /// # Errors
    ///
    /// As [`convert`](Self::convert), except that no unit is needed and nothing is written.
    pub fn probe(self) -> Result<FidelityReport, PvecConvertError> {
        let provider = AcadrustProvider::open(self.source).map_err(PvecConvertError::Open)?;
        let c = self.classify(&provider, Some(self.unit.unwrap_or(provider.source_unit())))?;
        c.tally
            .finish(c.encountered)
            .map_err(PvecConvertError::Tally)
    }

    /// As [`probe`](Self::probe), with provider `P`.
    ///
    /// # Errors
    ///
    /// As [`probe`](Self::probe), and [`PvecConvertError::UnitRequired`] when no unit was given and
    /// a paper sheet's plot paper units leave its frame to the drawing's unit, as
    /// [`convert_with`](Self::convert_with) would.
    pub fn probe_with<P: DwgProvider>(self) -> Result<FidelityReport, PvecConvertError> {
        let provider = P::open(self.source).map_err(PvecConvertError::Open)?;
        let c = self.classify(&provider, self.unit)?;
        c.tally
            .finish(c.encountered)
            .map_err(PvecConvertError::Tally)
    }

    /// Converts with the default provider, acadrust. The unit is the header's unless
    /// [`unit`](Self::unit) was set.
    ///
    /// # Errors
    ///
    /// [`PvecConvertError`]. After any error the destination and its report are as they were.
    pub fn convert(self) -> Result<PvecConversionOutput, PvecConvertError> {
        let observer = self.observer.clone();
        let result = if self.cancel.is_cancelled() {
            Err(PvecConvertError::Cancelled)
        } else {
            AcadrustProvider::open(self.source)
                .map_err(PvecConvertError::Open)
                .and_then(|provider| {
                    let unit = self.unit.unwrap_or_else(|| provider.source_unit());
                    self.run(&provider, unit)
                })
        };
        report_failure(observer.as_deref(), result)
    }

    /// Converts with provider `P`. The trait carries no unit, so [`unit`](Self::unit) is required.
    ///
    /// # Errors
    ///
    /// [`PvecConvertError::UnitRequired`] without a unit, otherwise as [`convert`](Self::convert).
    pub fn convert_with<P: DwgProvider>(self) -> Result<PvecConversionOutput, PvecConvertError> {
        let observer = self.observer.clone();
        let result = if self.cancel.is_cancelled() {
            Err(PvecConvertError::Cancelled)
        } else {
            self.unit
                .ok_or(PvecConvertError::UnitRequired)
                .and_then(|unit| {
                    let provider = P::open(self.source).map_err(PvecConvertError::Open)?;
                    self.run(&provider, unit)
                })
        };
        report_failure(observer.as_deref(), result)
    }

    /// Everything a conversion decides about its entities, up to but not including the write.
    /// [`probe`](Self::probe) stops here, so it and [`convert`](Self::convert) cannot disagree.
    ///
    /// `unit` is the drawing's unit, when there is one; a paper sheet whose plot settings name
    /// inches or millimetres is in that unit instead (P-6, #1562).
    fn classify<P: DwgProvider>(
        &self,
        provider: &P,
        unit: Option<SourceUnit>,
    ) -> Result<Classified, PvecConvertError> {
        let Self {
            source,
            view_profile,
            cancel,
            gate,
            observer,
            queue_capacity,
            components,
            regions,
            calibration: caller_calibration,
            ..
        } = self;
        let observer = observer.as_deref();
        let tell = |event: EngineEvent| {
            if let Some(o) = observer {
                o.on_event(event);
            }
        };
        tell(EngineEvent::ConversionSourceOpened {
            provider: provider.name().to_owned(),
        });
        if cancel.is_cancelled() {
            return Err(PvecConvertError::Cancelled);
        }
        let view = view_profile
            .apply(provider)
            .map_err(PvecConvertError::View)?;
        tell(EngineEvent::ConversionViewResolved {
            included: view.report.included,
            excluded: view.excluded.len() as u64,
        });
        if cancel.is_cancelled() {
            return Err(PvecConvertError::Cancelled);
        }
        let layouts = provider
            .layouts()
            .map_err(|e| PvecConvertError::View(e.into()))?;
        let layout =
            select_layout(&layouts, &view_profile.space).map_err(PvecConvertError::View)?;

        // The paper sheet (#1562): its frame and unit (P-6), its print flags (P-8) and the plot
        // scale as the calibration (P-9). Zero or missing plot settings are not trusted for any of
        // it: the frame is then the kept geometry's extent, in the drawing's unit, and nothing is
        // cut.
        let mut sheet: Option<Sheet> = None;
        if layout.kind == LayoutKind::Paper {
            let fact = match provider.layout_plot(layout.id) {
                Ok(f) => f,
                Err(ProviderError::CannotReport(_)) => SourceFact::NotReportable,
                Err(e) => return Err(PvecConvertError::View(e.into())),
            };
            let text = |v: f64| format!("{v}");
            let mut summary = SheetSummary {
                layout: layout.name.clone(),
                frame: "extents".into(),
                unit_source: "drawing".into(),
                calibration: if caller_calibration.is_some() {
                    "caller".into()
                } else {
                    "none".into()
                },
                ..SheetSummary::default()
            };
            let mut plot_unit = None;
            let mut frame_box = None;
            let mut drop_widths = false;
            let mut written = None;
            match &fact {
                SourceFact::Absent => summary.frame_reason = Some("plot-settings-absent".into()),
                SourceFact::Present(p) => {
                    summary.plot = Some(SheetPlotSummary {
                        paper_size: p.paper_size.clone(),
                        paper_width: text(p.paper_width),
                        paper_height: text(p.paper_height),
                        paper_units: p.paper_units,
                        rotation: p.rotation,
                        plot_type: p.plot_type,
                        scale_numerator: text(p.scale_numerator),
                        scale_denominator: text(p.scale_denominator),
                        style_sheet: p.style_sheet.clone(),
                        print_lineweights: p.print_lineweights,
                        scale_lineweights: p.scale_lineweights,
                        plot_plot_styles: p.plot_plot_styles,
                    });
                    let sized = |v: f64| v.is_finite() && v > 0.0;
                    if !(sized(p.paper_width) && sized(p.paper_height)) {
                        summary.frame_reason = Some("paper-size-zero".into());
                    } else {
                        let sheet_unit = match p.paper_units {
                            0 => Some(SourceUnit::Inches),
                            1 => Some(SourceUnit::Millimeters),
                            // The drawing's unit stands in. `convert_with` refuses without one, so
                            // the probes do too rather than frame the sheet differently.
                            _ => {
                                summary.notes.push("paper-units-fallback".into());
                                Some(unit.ok_or(PvecConvertError::UnitRequired)?)
                            }
                        };
                        // Millimetres of paper in the sheet unit: exact for millimetres, one
                        // division for inches, through PVec's own metres otherwise.
                        let to_unit = |mm: f64| match sheet_unit {
                            Some(SourceUnit::Millimeters) => Some(mm),
                            Some(SourceUnit::Inches) => Some(mm / 25.4),
                            Some(u) => metres_per_unit(u).ok().map(|m| mm * 0.001 / m),
                            None => None,
                        };
                        let (w, h) = if matches!(p.rotation, 1 | 3) {
                            (p.paper_height, p.paper_width)
                        } else {
                            (p.paper_width, p.paper_height)
                        };
                        // UNMEASURED (the print profile ADR's open item): the paper's lower left
                        // is taken to sit at -(margin + plot origin) from the paper-space origin.
                        // No fixture with non-zero margins and origin has measured it, so the
                        // report says `position: unmeasured`.
                        let placed = (
                            to_unit(p.margins[0] + p.origin[0]),
                            to_unit(p.margins[1] + p.origin[1]),
                            to_unit(w),
                            to_unit(h),
                        );
                        match placed {
                            (Some(dx), Some(dy), Some(w), Some(h)) => {
                                let (x0, y0) = (0.0 - dx, 0.0 - dy);
                                frame_box = Some([x0, y0, x0 + w, y0 + h]);
                                plot_unit = sheet_unit;
                            }
                            _ => summary.frame_reason = Some("unit-not-declarable".into()),
                        }
                    }
                    if let (Some(b), Some(sheet_unit)) = (frame_box, plot_unit) {
                        summary.frame = "paper".into();
                        summary.position = Some("unmeasured".into());
                        summary.sheet_box = Some(b.map(text));
                        if matches!(p.paper_units, 0 | 1) {
                            summary.unit_source = "plot-paper-units".into();
                        }
                        drop_widths = !p.print_lineweights;
                        summary.widths_dropped = drop_widths;
                        if p.plot_type != 5 {
                            summary.notes.push("plot-type-not-applied".into());
                        }
                        if p.rotation != 0 {
                            summary.notes.push("rotation-not-applied".into());
                        }
                        // P-9: the plot scale is the calibration record, never applied to a
                        // coordinate. A caller's calibration wins.
                        let (n, d) = (p.scale_numerator, p.scale_denominator);
                        if caller_calibration.is_some() {
                            summary.notes.push("caller-calibration".into());
                        } else if !(sized(n) && sized(d)) {
                            summary.notes.push("scale-zero".into());
                        } else {
                            // Both sides integral after scaling by one power of ten up to 10^6
                            // reduce exactly; anything else is rounded at 10^6 and the record
                            // carries the rounding as its uncertainty. A side is integral at `k`
                            // when its rounded multiple divided back by `k` is the stored value:
                            // a decimal such as 2.01 is no binary fraction, so `2.01 * 100` is not
                            // 201 in f64, but `201 / 100` is the f64 nearest 2.01, which is it.
                            let integral = |v: f64| (1.0..9.0e15).contains(&v).then_some(v as u64);
                            let at = |v: f64, k: f64| {
                                let r = (v * k).round();
                                if r / k == v { integral(r) } else { None }
                            };
                            let exact = [1.0, 10.0, 100.0, 1e3, 1e4, 1e5, 1e6]
                                .into_iter()
                                .find_map(|k| Some((at(n, k)?, at(d, k)?)));
                            let uncertain = exact.is_none();
                            let ratio = exact.or_else(|| {
                                Some((integral((n * 1e6).round())?, integral((d * 1e6).round())?))
                            });
                            let reduced = ratio.and_then(|(a, b)| {
                                let (mut x, mut y) = (a, b);
                                while y != 0 {
                                    (x, y) = (y, x % y);
                                }
                                Some((u32::try_from(a / x).ok()?, u32::try_from(b / x).ok()?))
                            });
                            let paper = metres_per_unit(sheet_unit).ok();
                            let claim = reduced.zip(paper).map(|((num, den), paper)| {
                                let raw = n / d;
                                let note = if uncertain {
                                    format!("plot scale {n}:{d} as stored, written as {num}:{den}")
                                } else {
                                    format!("plot scale {num}:{den}")
                                };
                                ScaleClaim {
                                    metres_per_source_unit: paper * f64::from(den) / f64::from(num),
                                    provenance: ScaleProvenance::PlanScale {
                                        numerator: num,
                                        denominator: den,
                                        paper_metres_per_unit: paper,
                                    },
                                    uncertainty: uncertain.then(|| {
                                        (f64::from(num) / f64::from(den) - raw).abs() / raw
                                    }),
                                    note: if note.len() <= 255 {
                                        note
                                    } else {
                                        format!("plot scale written as {num}:{den}")
                                    },
                                }
                            });
                            match claim.map(|c| Calibration::new(Some(c), None, None)) {
                                Some(Ok(c)) => {
                                    if uncertain {
                                        summary.notes.push("scale-not-reduced".into());
                                    }
                                    summary.calibration = if uncertain {
                                        "plot-scale-uncertain".into()
                                    } else {
                                        "plot-scale".into()
                                    };
                                    summary.scale = reduced.map(|(a, b)| format!("{a}:{b}"));
                                    written = Some(c);
                                }
                                _ => summary.notes.push("scale-refused".into()),
                            }
                        }
                    }
                }
                _ => summary.frame_reason = Some("plot-settings-not-reportable".into()),
            }
            summary.unit = plot_unit
                .or(unit)
                .map_or("Unknown", SourceUnit::name)
                .into();
            sheet = Some(Sheet {
                summary,
                frame: frame_box,
                unit: plot_unit,
                calibration: written,
                drop_widths,
            });
        }
        let source_name = match *source {
            CadSource::Path(p) => p.display().to_string(),
            CadSource::Bytes(_) => "(bytes)".to_owned(),
        };
        let flat = BlockResolver::new(provider, FlattenBounds::from_profile(view_profile))
            .flatten(
                &EntityScope::Layout(layout.id),
                &source_name,
                &layout.name,
                &mut NoProvenance,
                &mut LineageStore::new(),
            )
            .map_err(PvecConvertError::Flatten)?;

        // The leaves the profile kept. The view pass decides visibility and the flatten pass owns
        // the composed maps, so a flattened leaf is drawn when the view pass kept the same one.
        let kept: BTreeSet<(Option<u64>, String, String)> =
            view.included.iter().map(leaf_key).collect();
        drop(view.included);
        let excluded_inserts: BTreeSet<u64> = view
            .excluded
            .iter()
            .filter_map(|x| x.handle.map(|h| h.get()))
            .collect();

        let mut items: Vec<Item<'_>> = Vec::new();
        for f in &flat.entities {
            let p = &f.entity.provenance;
            if kept.contains(&(
                p.handle.map(|h| h.get()),
                p.block_path.to_string(),
                p.class_name.clone(),
            )) {
                items.push(Item::Leaf(f));
            }
        }
        drop(kept);
        items.extend(view.excluded.iter().map(Item::Excluded));
        for s in &flat.skipped {
            if s.insert_handle
                .is_some_and(|h| excluded_inserts.contains(&h.get()))
            {
                continue;
            }
            items.push(Item::Skipped(s));
        }

        // Ids are positions in the canonical order, which is decided by block path, handle and
        // arrival position alone (arrival positions are all different, so no later key is ever
        // reached). Positions are sorted by those keys read in place, so no second copy of an
        // entity is made to number them.
        let mut perm: Vec<usize> = (0..items.len()).collect();
        let key = |i: usize| match &items[i] {
            Item::Leaf(f) => (
                &f.entity.provenance.block_path,
                f.entity.provenance.handle,
                i as u64,
            ),
            Item::Excluded(x) => (&x.block_path, x.handle, i as u64),
            Item::Skipped(s) => (&s.block_path, s.insert_handle, i as u64),
        };
        perm.sort_by(|&a, &b| key_cmp(key(a), key(b)));

        // The classifying side produces, a consumer thread turns each outcome into a row, a
        // record, a diagnostic and a tally entry. They meet in a bounded queue, so a slow consumer
        // makes the producer wait instead of letting outcomes pile up.
        // The wall lines and the room names the regions step reads, in canonical order.
        let rules = regions.is_some().then(ElementRules::default_profile);
        let mut walls: Vec<WallEdge> = Vec::new();
        let mut labels: Vec<RoomLabel> = Vec::new();
        let (tx, rx) = bounded::<Outcome>(*queue_capacity);
        let sent = AtomicU64::new(0);
        /// The pieces of a line or straight polyline inside the sheet `b`, `[min x, min y, max x,
        /// max y]`, as point lists; `None` for a curve that is not cut (P-7). Cut on x, then on y,
        /// Liang-Barsky per segment: a crossing point is computed once, from its segment's own
        /// ends, and lies exactly on the edge it was cut on. A closed polyline's pieces are joined
        /// back across the point where it closed.
        fn sheet_cut(curve: &Curve, b: [f64; 4]) -> Option<Vec<Vec<[f64; 2]>>> {
            let points: Vec<[f64; 2]> = match curve {
                Curve::Polyline(p) => p.vertices().iter().map(|v| [v.x, v.y]).collect(),
                Curve::Trimmed(t) if matches!(t.basis(), Curve::Line(_)) => {
                    let (a, z) = (t.eval(0.0).ok()?, t.eval(t.trim().span).ok()?);
                    vec![[a.x, a.y], [z.x, z.y]]
                }
                _ => return None,
            };
            let mut pieces = vec![points.clone()];
            for (axis, lo, hi) in [(0, b[0], b[2]), (1, b[1], b[3])] {
                let mut out: Vec<Vec<[f64; 2]>> = Vec::new();
                for piece in &pieces {
                    let mut cur: Vec<[f64; 2]> = Vec::new();
                    for w in piece.windows(2) {
                        let (a, z) = (w[0], w[1]);
                        let (da, dz) = (a[axis], z[axis]);
                        // The part inside, as parameters and the edge each end was cut on.
                        let (mut t0, mut e0, mut t1, mut e1) = (0.0f64, None, 1.0f64, None);
                        if da == dz {
                            if da < lo || da > hi {
                                if cur.len() >= 2 {
                                    out.push(std::mem::take(&mut cur));
                                }
                                cur.clear();
                                continue;
                            }
                        } else {
                            let (ta, tb) = ((lo - da) / (dz - da), (hi - da) / (dz - da));
                            let ((enter, ee), (leave, le)) = if ta < tb {
                                ((ta, lo), (tb, hi))
                            } else {
                                ((tb, hi), (ta, lo))
                            };
                            if enter > t0 {
                                (t0, e0) = (enter, Some(ee));
                            }
                            if leave < t1 {
                                (t1, e1) = (leave, Some(le));
                            }
                            if t0 >= t1 {
                                if cur.len() >= 2 {
                                    out.push(std::mem::take(&mut cur));
                                }
                                cur.clear();
                                continue;
                            }
                        }
                        let at = |t: f64, edge: Option<f64>, end: [f64; 2]| match edge {
                            None => end,
                            Some(e) => {
                                let mut q = [a[0] + (z[0] - a[0]) * t, a[1] + (z[1] - a[1]) * t];
                                q[axis] = e;
                                q
                            }
                        };
                        let (p0, p1) = (at(t0, e0, a), at(t1, e1, z));
                        if cur.last() != Some(&p0) {
                            if cur.len() >= 2 {
                                out.push(std::mem::take(&mut cur));
                            }
                            cur.clear();
                            cur.push(p0);
                        }
                        cur.push(p1);
                        if e1.is_some() {
                            out.push(std::mem::take(&mut cur));
                        }
                    }
                    if cur.len() >= 2 {
                        out.push(cur);
                    }
                }
                pieces = out;
            }
            let closed = points.len() > 2 && points.first() == points.last();
            if closed
                && pieces.len() > 1
                && pieces[0].first() == points.first()
                && pieces[pieces.len() - 1].last() == points.last()
            {
                let first = pieces.remove(0);
                let last = pieces.len() - 1;
                pieces[last].extend_from_slice(&first[1..]);
                let joined = pieces.pop().expect("one piece at least");
                pieces.insert(0, joined);
            }
            Some(pieces)
        }
        let frame = sheet.as_ref().and_then(|s| s.frame);
        let drop_widths = sheet.as_ref().is_some_and(|s| s.drop_widths);
        // What the sheet clip did: outside, crossing, cut and the pieces the cut ones became.
        let mut clipped = [0u64; 4];
        let mut viewports = 0u64;
        // Ids are dense over what is written to the report, so a cut entity's pieces each take one.
        let mut next_id = 0u64;
        let consumed = std::thread::scope(|scope| {
            let consumer = scope.spawn(|| consume(rx, &sent, observer));
            let mut failure = None;
            for (n, &at) in perm.iter().enumerate() {
                if n.is_multiple_of(BATCH) && cancel.is_cancelled() {
                    failure = Some(PvecConvertError::Cancelled);
                    break;
                }
                // A leaf is classified where it sits; an excluded or skipped one has no entity of
                // its own, so its stand-in is made for the moment it is classified.
                let stand_in;
                let raw: &RawEntity = match &items[at] {
                    Item::Leaf(f) => &f.entity,
                    Item::Excluded(x) => {
                        stand_in = stub(
                            x.handle.map(|h| h.get()),
                            &x.class_name,
                            x.block_path.clone(),
                            &x.layer,
                            x.reason.code(),
                        );
                        &stand_in
                    }
                    Item::Skipped(s) => {
                        stand_in = stub(
                            s.insert_handle.map(|h| h.get()),
                            "INSERT",
                            s.block_path.clone(),
                            "",
                            s.reason.code(),
                        );
                        &stand_in
                    }
                };
                let mut done = match classify_item(&items[at], raw, gate) {
                    Ok(done) => done,
                    Err(e) => {
                        failure = Some(e);
                        break;
                    }
                };
                // A viewport on the sheet shows model space, which is not drawn yet (#1563): it is
                // reported, never dropped.
                if sheet.is_some()
                    && matches!(items[at], Item::Leaf(_))
                    && raw.provenance.class_name == "VIEWPORT"
                {
                    viewports += 1;
                    done = without_geometry(
                        Classification::refused(ReasonCode::ViewNotEnabled, "VIEWPORT"),
                        vec![
                            "VIEW_NOT_ENABLED (508): model space through a viewport is not drawn yet (#1563)"
                                .to_owned(),
                        ],
                    );
                }
                // P-7: nothing is drawn outside the sheet. A line or a straight polyline is cut
                // exactly and may become several pieces, each its own entity and row; any other
                // curve is kept when wholly inside, dropped when wholly outside and refused when
                // it crosses the edge. PVec's own box decides inside and outside.
                let mut more: Vec<Done> = Vec::new();
                let mut whole = true;
                if let (Some(b), Some(bounds), Some(curve)) = (
                    frame,
                    done.geometry.source_bounds(),
                    done.geometry.geometry(),
                ) {
                    let (lo, hi) = (bounds.min(), bounds.max());
                    let class = raw.provenance.class_name.as_str();
                    if hi.x < b[0] || lo.x > b[2] || hi.y < b[1] || lo.y > b[3] {
                        clipped[0] += 1;
                        whole = false;
                        done = without_geometry(
                            Classification::ignored_by_policy("viewport_clipping", class),
                            vec![format!(
                                "{}: wholly outside the sheet",
                                PolicyReason::OutsideViewport.code()
                            )],
                        );
                    } else if !(lo.x >= b[0] && hi.x <= b[2] && lo.y >= b[1] && hi.y <= b[3]) {
                        whole = false;
                        match sheet_cut(curve, b) {
                            None => {
                                clipped[1] += 1;
                                done = without_geometry(
                                    Classification::refused(ReasonCode::ViewNotEnabled, class),
                                    vec![
                                        "VIEW_NOT_ENABLED (508): the curve crosses the sheet edge, and a curve is not cut until curve splitting lands (#1565)"
                                            .to_owned(),
                                    ],
                                );
                            }
                            Some(pieces) if pieces.is_empty() => {
                                clipped[0] += 1;
                                done = without_geometry(
                                    Classification::ignored_by_policy("viewport_clipping", class),
                                    vec![format!(
                                        "{}: wholly outside the sheet",
                                        PolicyReason::OutsideViewport.code()
                                    )],
                                );
                            }
                            Some(pieces) => {
                                let line = matches!(curve, Curve::Trimmed(_));
                                let fidelity = done.geometry.fidelity();
                                let built: Result<Vec<Normalized2DGeometry>, String> = pieces
                                    .iter()
                                    .map(|p| {
                                        if line {
                                            let a = Point2::checked(p[0][0], p[0][1])
                                                .map_err(|e| e.to_string())?;
                                            let z = Point2::checked(p[1][0], p[1][1])
                                                .map_err(|e| e.to_string())?;
                                            let l = Line::new(a, a.vector_to(z))
                                                .map_err(|e| e.to_string())?;
                                            let t = Trimmed::new(
                                                Curve::Line(l),
                                                0.0,
                                                a.distance(z),
                                                true,
                                            )
                                            .map_err(|e| e.to_string())?;
                                            Normalized2DGeometry::from_curve(
                                                Curve::Trimmed(t),
                                                fidelity,
                                            )
                                            .map_err(|e| e.to_string())
                                        } else {
                                            Normalized2DGeometry::polyline(p, fidelity)
                                                .map_err(|e| e.to_string())
                                        }
                                    })
                                    .collect();
                                match built {
                                    Err(e) => {
                                        done = without_geometry(
                                            Classification::refused(
                                                ReasonCode::DegenerateGeometry,
                                                class,
                                            ),
                                            vec![format!(
                                                "{class}: a piece cut at the sheet edge is degenerate: {e}"
                                            )],
                                        );
                                    }
                                    Ok(records) => {
                                        clipped[2] += 1;
                                        clipped[3] += records.len() as u64;
                                        let mut gated_pieces = Vec::with_capacity(records.len());
                                        for r in records {
                                            match gated(r, &raw.provenance, gate) {
                                                Ok(d) => gated_pieces.push(d),
                                                Err(e) => {
                                                    failure = Some(e);
                                                    break;
                                                }
                                            }
                                        }
                                        if failure.is_some() {
                                            break;
                                        }
                                        let mut it = gated_pieces.into_iter();
                                        if let Some(first) = it.next() {
                                            done = first;
                                        }
                                        more = it.collect();
                                    }
                                }
                            }
                        }
                    }
                }
                let mut stop = false;
                for done in std::iter::once(done).chain(more) {
                    let mut diagnostic =
                        entity_diagnostic(&items[at], raw, &done.classification, &layout.name);
                    if let Some(d) = &mut diagnostic {
                        d.source = Some(source_name.clone());
                    }
                    let id = EntityId::at(next_id);
                    next_id += 1;
                    if let (Some(rules), Item::Leaf(f), true) = (&rules, &items[at], whole) {
                        match &raw.geometry {
                            RawGeometry::Line { start, end }
                                if done.geometry.geometry().is_some()
                                    && rules.classify(raw).class == ElementClass::Wall =>
                            {
                                walls.push(WallEdge {
                                    entity: id,
                                    start: f.map_point(*start),
                                    end: f.map_point(*end),
                                });
                            }
                            RawGeometry::Unmapped { .. }
                                if matches!(
                                    raw.provenance.class_name.as_str(),
                                    "TEXT" | "MTEXT"
                                ) =>
                            {
                                let anchor =
                                    raw.provenance.handle.and_then(|h| provider.text_anchor(h));
                                if let Some((text, at)) =
                                    anchor.filter(|(t, _)| !t.trim().is_empty())
                                {
                                    labels.push(RoomLabel {
                                        at: f.map_point(at),
                                        text: text.trim().to_owned(),
                                    });
                                }
                            }
                            _ => {}
                        }
                    }
                    let side = (*components != StreamSelection::NONE
                        && done.geometry.geometry().is_some())
                    .then(|| {
                        // Counted per kept entity in `open`, once the writer has said which it kept.
                        let (style, colour, bad_style) = if components.line_style {
                            let mut input = dwg_style_input(&raw.style, true);
                            // P-8: a layout that does not print lineweights has none on its sheet.
                            if drop_widths {
                                input.lineweight = None;
                            }
                            let c = dwg_line_style(id, &input).ok();
                            let bad = c.is_none();
                            (c, input.colour_index.is_some(), bad)
                        } else {
                            (None, false, false)
                        };
                        Side {
                            // Only the handle, the layer and the outermost INSERT are read; the class name and
                            // the rest of the block path are in the report, so they are not kept per entity.
                            stub: ProvenanceStub {
                                handle: raw.provenance.handle,
                                class_name: String::new(),
                                block_path: raw.provenance.block_path.steps().first().map_or_else(
                                    crate::pvec::InstancePath::root,
                                    |s| {
                                        crate::pvec::InstancePath::root()
                                            .child(String::new(), s.insert_handle)
                                    },
                                ),
                                layer: raw.provenance.layer.clone(),
                            },
                            style,
                            colour,
                            bad_style,
                        }
                    });
                    let outcome = Outcome {
                        id,
                        handle: raw.provenance.handle.map(|h| h.get()),
                        done,
                        diagnostic,
                        side,
                    };
                    if tx.send(outcome).is_err() {
                        stop = true;
                        break;
                    }
                    sent.fetch_add(1, Ordering::Release);
                }
                if stop {
                    break;
                }
            }
            drop(tx);
            let consumed = consumer.join().expect("the consumer thread does not panic");
            failure.map_or(Ok(consumed), Err)
        })?;
        let encountered = next_id;
        let sheet = sheet.map(|mut s| {
            for (key, n) in ["outside", "crossing", "cut", "pieces"]
                .into_iter()
                .zip(clipped)
            {
                s.summary.clip.insert(key.to_owned(), n);
            }
            s.summary.viewports = viewports;
            s
        });
        let xrefs = XrefSummary {
            unresolved: flat
                .skipped
                .iter()
                .filter(|s| s.reason == SkipReason::ExternalReference)
                .map(|s| UnresolvedXrefRow {
                    name: s.block_name.clone(),
                    path: String::new(),
                    resolved_path: None,
                    path_refusal: None,
                })
                .collect(),
            resolved: Vec::new(),
        };
        Ok(Classified {
            geometries: consumed.geometries,
            stored: consumed.stored,
            rows: consumed.rows,
            tally: consumed.tally,
            encountered,
            view: ViewSummary::from_report(&view.report),
            xrefs,
            diagnostics: consumed.diagnostics,
            peak_queue_depth: consumed.peak,
            sides: consumed.sides,
            walls,
            labels,
            sheet,
        })
    }

    fn run<P: DwgProvider>(
        self,
        provider: &P,
        unit: SourceUnit,
    ) -> Result<PvecConversionOutput, PvecConvertError> {
        let Classified {
            geometries,
            stored,
            mut rows,
            mut tally,
            encountered,
            view,
            xrefs,
            diagnostics,
            peak_queue_depth,
            sides,
            walls,
            labels,
            sheet,
        } = self.classify(provider, Some(unit))?;
        let Self {
            dest,
            leaves,
            spill,
            cancel,
            gate,
            calibration,
            observer,
            components: select,
            regions,
            ..
        } = self;
        // A sheet is in the unit its plot settings name, its box is the paper (so the writer
        // never refuses for bounds), and its plot scale is the calibration unless the caller gave
        // one (#1562).
        let (unit, area, calibration, sheet) = match sheet {
            Some(s) => (
                s.unit.unwrap_or(unit),
                s.frame,
                calibration.or(s.calibration),
                Some(s.summary),
            ),
            None => (unit, None, calibration, None),
        };
        let tell = |event: EngineEvent| {
            if let Some(o) = &observer {
                o.on_event(event);
            }
        };

        let write_error = |e: WriteError| match e {
            WriteError::Cancelled => PvecConvertError::Cancelled,
            other => PvecConvertError::Write(other),
        };
        // What both paths do once the artifact is written: tell the observer, close the report against
        // what went in. It runs once, so it takes the rows.
        // With no `written` (PVec refused every curve) there is no artifact, so the report is closed
        // against nothing written and the observer is not told.
        // The tally is sealed here and not at classification: an entity the classifier called
        // supported can still be one PVec's container refuses, and it is counted as that.
        let seal = |written: Option<(&BoundsReport, u64, u64)>,
                    refused: &[(EntityId, String)],
                    components: Option<ComponentsSummary>,
                    found: Option<RegionsSummary>|
         -> Result<(ConversionReport, FidelityReport), PvecConvertError> {
            for (id, rule) in refused {
                let from = &stored
                    .iter()
                    .find(|(s, _)| s == id)
                    .expect("a refused entity was written from a stored classification")
                    .1;
                let to = Classification::refused(ReasonCode::PvecCannotStore, from.source_class());
                tally.reclassify(from, &to);
                rows.refuse(*id, &to, format!("PVec's writer refused it ({rule})"));
            }
            let fidelity = tally.finish(encountered).map_err(PvecConvertError::Tally)?;
            if let Some((_, entities, bytes)) = written {
                tell(EngineEvent::ConversionArtifactWritten { entities, bytes });
            }
            let mut report = rows
                .finish(
                    encountered,
                    view,
                    xrefs,
                    BoundsSplit {
                        exact: written.map_or(0, |(b, ..)| b.exact_count() as u64),
                        hull_upper_bound: written.map_or(0, |(b, ..)| b.hull_count() as u64),
                    },
                )
                .map_err(PvecConvertError::Report)?;
            report.capabilities = Some(gate.summary());
            report.diagnostics = diagnostics;
            report.components = components;
            report.regions = found;
            report.sheet = sheet.clone();
            Ok((report, fidelity))
        };
        let config = WriteConfig { unit, leaves };
        let mut chunk = 0u64;
        let mut on_chunk = |entities: u64, bytes: u64| {
            chunk += 1;
            tell(EngineEvent::ConversionChunkWritten {
                chunk,
                entities,
                bytes,
            });
        };
        // The geometry is written first, then the streams against the ids the writer kept, because a
        // refused curve moves every id after it. The streams go in after the geometry, so they cannot
        // change a geometry block.
        // The writer takes what it keeps out of the geometry, so the geometry goes the moment the
        // open call returns, whether it succeeded or not, and the report is built without it.
        let open = |geometries: Vec<(EntityId, Normalized2DGeometry)>,
                    spill: Option<&SpillConfig>,
                    on_chunk: &mut dyn FnMut(u64, u64)|
         -> Result<
            (
                OpenArtifact,
                Option<ComponentsSummary>,
                Option<RegionsSummary>,
            ),
            PvecConvertError,
        > {
            let opened = open_geometry(
                &config,
                geometries.iter().map(|(id, g)| (*id, g)),
                Vec::new(),
                area,
                spill,
                &cancel,
                on_chunk,
            );
            drop(geometries);
            let mut open = opened.map_err(write_error)?;
            // PVec sorts the calibration block after every other kind, so it can go in first.
            if let Some(c) = &calibration {
                open.calibration(c).map_err(write_error)?;
            }
            if select == StreamSelection::NONE && regions.is_none() {
                return Ok((open, None, None));
            }
            let mut streams = Vec::new();
            let mut summary = None;
            if select != StreamSelection::NONE {
                let layers = if select.layer {
                    provider
                        .layers()
                        .map_err(|e| PvecConvertError::View(e.into()))?
                } else {
                    Vec::new()
                };
                let rows: Vec<ComponentSource<'_>> = sides
                    .iter()
                    .map(|(id, s)| ComponentSource {
                        id: *id,
                        stub: &s.stub,
                        style: s.style.as_ref(),
                    })
                    .collect();
                let built = build_streams(select, &open.source_ids, &layers, &rows)
                    .map_err(PvecConvertError::Components)?;
                // Only the entities the writer kept: one it refused has no row, so it is not counted here.
                let kept: BTreeSet<EntityId> = open.source_ids.iter().copied().collect();
                let kept_sides = || {
                    sides
                        .iter()
                        .filter(|(id, _)| kept.contains(id))
                        .map(|(_, s)| s)
                };
                let mut made = ComponentsSummary {
                    streams: built.names.iter().map(|n| (*n).to_owned()).collect(),
                    dropped: BTreeMap::new(),
                    colours_ignored: kept_sides().filter(|s| s.colour).count() as u64,
                };
                let style_bad = kept_sides().filter(|s| s.bad_style).count() as u64;
                for d in &built.dropped {
                    *made
                        .dropped
                        .entry(format!("{}.{}", d.stream, d.reason))
                        .or_default() += 1;
                }
                if style_bad > 0 {
                    made.dropped
                        .insert("line-style.bad-style".to_owned(), style_bad);
                }
                streams = built.streams;
                summary = Some(made);
            }
            let mut found = None;
            if let Some(config) = &regions {
                let mut report = RegionsSummary::default();
                match derive_regions(&walls, &labels, &[], config) {
                    // A plan past a limit converts without regions, and the report says which limit.
                    Err(e) => report.refused = Some(e.reason_code().to_owned()),
                    Ok(derived) => {
                        report.rooms = derived.report.rooms;
                        report.corridors = derived.report.corridors;
                        report.gaps_closed = derived.report.gaps_closed;
                        report.unlabelled = derived.report.unlabelled;
                        report.labelled =
                            derived.regions.iter().filter(|r| r.label.is_some()).count() as u64;
                        let class = semantic_stream(&derived, &open.source_ids)
                            .map_err(PvecConvertError::Components)?;
                        for d in &class.dropped {
                            *report.dropped.entry(d.reason.to_owned()).or_default() += 1;
                        }
                        report.written = (derived.regions.len() - class.dropped.len()) as u64;
                        streams.extend(class.stream);
                    }
                }
                found = Some(report);
            }
            open.components(streams, &cancel).map_err(write_error)?;
            Ok((open, summary, found))
        };
        // The report is written from the sealed value straight into its partial file, so the whole
        // of it never exists as one string. A failure while writing it reaches the caller as
        // `Publish(Io)`, the publisher's own error, not as `Report`.
        let write_report = |sealed: &(ConversionReport, FidelityReport, u64),
                            out: &mut dyn std::io::Write| {
            sealed.0.write_json(out).map_err(std::io::Error::other)
        };
        let publish_error = |e: StreamedPublishError<PvecConvertError>| match e {
            StreamedPublishError::Produce(e) => e,
            StreamedPublishError::Publish(PublishError::Cancelled) => PvecConvertError::Cancelled,
            StreamedPublishError::Publish(other) => PvecConvertError::Publish(other),
        };
        let published = match &spill {
            None => match open(geometries, None, &mut on_chunk) {
                Err(PvecConvertError::Write(WriteError::AllRefused(refused))) => {
                    seal(None, &refused, None, None).and_then(|(report, fidelity)| {
                        Err(PvecConvertError::NothingStorable {
                            report: Box::new(report),
                            fidelity: Box::new(fidelity),
                        })
                    })
                }
                opened => opened.and_then(|(open, summary, found)| {
                    let artifact = open.finish().map_err(write_error)?;
                    publish_streamed_with(
                        dest,
                        &cancel,
                        &mut |_| {},
                        |out| {
                            out.write_all(&artifact.bytes)
                                .map_err(|e| PvecConvertError::Publish(PublishError::Io(e)))?;
                            let (report, fidelity) = seal(
                                Some((
                                    &artifact.bounds,
                                    artifact.written,
                                    artifact.bytes.len() as u64,
                                )),
                                &artifact.refused,
                                summary,
                                found,
                            )?;
                            Ok((report, fidelity, artifact.written))
                        },
                        write_report,
                    )
                    .map_err(publish_error)
                }),
            },
            Some(spill) => publish_streamed_with(
                dest,
                &cancel,
                &mut |_| {},
                |out| {
                    let (open, summary, found) = match open(geometries, Some(spill), &mut on_chunk)
                    {
                        Err(PvecConvertError::Write(WriteError::AllRefused(refused))) => {
                            let (report, fidelity) = seal(None, &refused, None, None)?;
                            return Err(PvecConvertError::NothingStorable {
                                report: Box::new(report),
                                fidelity: Box::new(fidelity),
                            });
                        }
                        opened => opened?,
                    };
                    let artifact = open
                        .seal_into(&cancel, &mut on_chunk, out)
                        .map_err(write_error)?;
                    tell(EngineEvent::ConversionSpoolSpilled {
                        bytes: artifact.length,
                    });
                    let (report, fidelity) = seal(
                        Some((&artifact.bounds, artifact.written, artifact.length)),
                        &artifact.refused,
                        summary,
                        found,
                    )?;
                    Ok((report, fidelity, artifact.written))
                },
                write_report,
            )
            .map_err(publish_error),
        };
        let (report, fidelity, written) = published?;
        tell(EngineEvent::ConversionPublished);
        Ok(PvecConversionOutput {
            artifact: dest.to_path_buf(),
            report,
            fidelity,
            written,
            peak_queue_depth,
        })
    }
}

/// What `classify` hands on: one record and one row per entity, and the counts that check them.
struct Classified {
    geometries: Vec<(EntityId, Normalized2DGeometry)>,
    /// The classification of each entity in `geometries`, for one the writer then refuses.
    stored: Vec<(EntityId, Classification)>,
    rows: ReportBuilder,
    tally: Tally,
    encountered: u64,
    view: ViewSummary,
    xrefs: XrefSummary,
    diagnostics: Vec<Diagnostic>,
    peak_queue_depth: u32,
    /// What the component streams need of each written entity; empty when they are off.
    sides: Vec<(EntityId, Side)>,
    /// The straight wall leaves and the text anchors the regions step reads; empty when it is off.
    walls: Vec<WallEdge>,
    labels: Vec<RoomLabel>,
    /// The paper sheet, for a paper layout under the print profile.
    sheet: Option<Sheet>,
}

/// A paper sheet as `classify` worked it out (#1562): the report's section, the frame to clip to
/// and to hand the writer, the unit the plot settings put the sheet in, the calibration the plot
/// scale gives and whether widths print.
struct Sheet {
    summary: SheetSummary,
    /// `[min x, min y, max x, max y]` in the sheet unit, `None` when the frame is the extents.
    frame: Option<[f64; 4]>,
    unit: Option<SourceUnit>,
    calibration: Option<Calibration>,
    drop_widths: bool,
}

/// What the component streams read off one written entity: where it came from and its line style.
struct Side {
    stub: ProvenanceStub,
    style: Option<LineStyleComponent>,
    /// The entity carries a DWG colour index (read only when the line style stream is on).
    colour: bool,
    /// The line style stream is on and the DWG facts could not make a component of this entity.
    bad_style: bool,
}

/// One entity's outcome on its way from the classifier to the consumer.
struct Outcome {
    id: EntityId,
    handle: Option<u64>,
    done: Done,
    diagnostic: Option<Diagnostic>,
    side: Option<Side>,
}

/// What the consumer built from the outcomes.
struct Consumed {
    tally: Tally,
    rows: ReportBuilder,
    geometries: Vec<(EntityId, Normalized2DGeometry)>,
    stored: Vec<(EntityId, Classification)>,
    diagnostics: Vec<Diagnostic>,
    sides: Vec<(EntityId, Side)>,
    peak: u32,
}

/// Drains the queue: tallies, rows, records and diagnostics, and the observer's events in order.
/// The depth is sampled after each receive from the sends that completed, so it can read one low
/// and never one high.
fn consume(
    rx: impl IntoIterator<Item = Outcome>,
    sent: &AtomicU64,
    observer: Option<&dyn EngineObserver>,
) -> Consumed {
    let mut out = Consumed {
        tally: Tally::new(),
        rows: ReportBuilder::new(),
        geometries: Vec::new(),
        stored: Vec::new(),
        diagnostics: Vec::new(),
        sides: Vec::new(),
        peak: 0,
    };
    let mut received = 0u64;
    for Outcome {
        id,
        handle,
        done,
        diagnostic,
        side,
    } in rx
    {
        received += 1;
        let depth = sent.load(Ordering::Acquire).saturating_sub(received);
        out.peak = out.peak.max(u32::try_from(depth).unwrap_or(u32::MAX));
        out.tally.record(&done.classification);
        if let Some(d) = diagnostic {
            if let Some(o) = observer {
                o.on_event(diagnostic_event(&d));
            }
            out.diagnostics.push(d);
        }
        out.rows.push(EntityRow::new(
            id,
            &done.classification,
            handle,
            done.geometry
                .geometry()
                .and_then(|_| kind_of(&done.geometry))
                .map(|k| row_for(k).pvec_type),
            done.warnings,
        ));
        if let Some(side) = side {
            out.sides.push((id, side));
        }
        if done.geometry.geometry().is_some() {
            out.stored.push((id, done.classification));
        }
        out.geometries.push((id, done.geometry));
        if received.is_multiple_of(BATCH as u64)
            && let Some(o) = observer
        {
            o.on_event(EngineEvent::ConversionBatchNormalized { entities: received });
        }
    }
    out
}

/// The capability gate: which capabilities pass, whether only exact entities do, and what happens
/// to the rest. The three vary together per profile.
struct Gate {
    name: &'static str,
    allowed: BTreeSet<CapabilityId>,
    exact_only: bool,
    policy: CapabilityPolicy,
}

impl Gate {
    fn of(profile: ConversionProfile) -> Self {
        Self {
            name: profile.name(),
            allowed: profile.capabilities(),
            exact_only: profile.exact_only(),
            policy: profile.policy(),
        }
    }

    fn summary(&self) -> CapabilitySummary {
        CapabilitySummary {
            profile: self.name.to_owned(),
            policy: format!("{:?}", self.policy),
            exact_only: self.exact_only,
            allowed: self.allowed.iter().map(|c| c.get()).collect(),
        }
    }
}

/// What the conversion met: a drawn leaf, a leaf the profile excluded, or an INSERT that was not
/// expanded.
enum Item<'a> {
    Leaf(&'a FlatEntity),
    Excluded(&'a ExcludedEntity),
    Skipped(&'a SkippedInsert),
}

/// One entity's outcome: its class, its record (with geometry when it has any) and the warnings.
struct Done {
    classification: Classification,
    geometry: Normalized2DGeometry,
    warnings: Vec<String>,
}

fn leaf_key(v: &ViewedEntity) -> (Option<u64>, String, String) {
    let p = &v.entity.provenance;
    (
        p.handle.map(|h| h.get()),
        p.block_path.to_string(),
        p.class_name.clone(),
    )
}

/// A stand-in `RawEntity` for something that has no geometry of its own, so it can be ordered and
/// numbered with the rest.
fn stub(
    handle: Option<u64>,
    class_name: &str,
    block_path: crate::pvec::InstancePath,
    layer: &str,
    why: &str,
) -> RawEntity {
    RawEntity {
        style: EntityStyle::absent(),
        geometry: RawGeometry::Unmapped {
            reason: why.to_owned(),
        },
        provenance: ProvenanceStub {
            handle: handle.map(crate::cad::ItemHandle::new),
            class_name: class_name.to_owned(),
            block_path,
            layer: layer.to_owned(),
        },
    }
}

fn without_geometry(classification: Classification, warnings: Vec<String>) -> Done {
    let geometry = Normalized2DGeometry::without_geometry(classification.class())
        .expect("a refusing class carries no geometry");
    Done {
        classification,
        geometry,
        warnings,
    }
}

fn classify_item(item: &Item<'_>, raw: &RawEntity, gate: &Gate) -> Result<Done, PvecConvertError> {
    let class = raw.provenance.class_name.as_str();
    Ok(match item {
        Item::Excluded(x) => without_geometry(x.reason.classification(class), Vec::new()),
        Item::Skipped(s) => {
            let reason = match s.reason {
                SkipReason::NormalNotZ => ReasonCode::ExtrusionNotApplied,
                _ => ReasonCode::UnresolvedBlock,
            };
            without_geometry(
                Classification::refused(reason, "INSERT"),
                vec![format!("INSERT of `{}`: {}", s.block_name, s.reason.code())],
            )
        }
        Item::Leaf(f) => classify_leaf(f, gate)?,
    })
}

fn classify_leaf(f: &FlatEntity, gate: &Gate) -> Result<Done, PvecConvertError> {
    let class = f.entity.provenance.class_name.as_str();
    let basic = match dwg_basic::extract(&f.entity) {
        Ok(b) => b,
        Err(BasicError::NotBasic) => return classify_spline_leaf(f, gate),
        Err(e) => {
            return Ok(without_geometry(
                e.classification(class),
                vec![e.warning(class)],
            ));
        }
    };
    let record = match Normalized2DGeometry::from_curve(basic.curve, basic.classification.class()) {
        Ok(r) => r,
        Err(e) => {
            return Ok(without_geometry(
                Classification::refused(ReasonCode::DegenerateGeometry, class),
                vec![format!("{class}: {e}")],
            ));
        }
    };
    promote_leaf(f, record, gate)
}

/// Pushes `record` through the map flattening composed for its leaf, and checks it against the
/// gate. A leaf a layout owns directly has the identity map.
fn promote_leaf(
    f: &FlatEntity,
    record: Normalized2DGeometry,
    gate: &Gate,
) -> Result<Done, PvecConvertError> {
    let class = f.entity.provenance.class_name.as_str();
    // A leaf a layout owns directly has the identity map. A nested one is pushed through the map
    // flattening composed for it, as one level named by its innermost INSERT.
    let path = &f.entity.provenance.block_path;
    let prov = &f.entity.provenance;
    let Some(step) = path.steps().last().cloned() else {
        return gated(record, prov, gate);
    };
    let composed = match ComposedTransform::identity().push_inner_coefficients(step, f.matrix()) {
        Ok(c) => c,
        Err(e) => {
            return Ok(without_geometry(
                e.classification(class),
                vec![e.to_string()],
            ));
        }
    };
    match promote(&record, &composed) {
        Ok(p) if p.record().geometry().is_some() => gated(p.record().clone(), prov, gate),
        Ok(p) => {
            let reason = p
                .reason()
                .map_or(ReasonCode::UnmappedSourceClass, |r| r.reason_code());
            Ok(without_geometry(
                Classification::refused(reason, class),
                Vec::new(),
            ))
        }
        Err(e) => Ok(without_geometry(
            Classification::refused(ReasonCode::UnmappedSourceClass, class),
            vec![format!("{class}: {e:?}")],
        )),
    }
}

/// A leaf the basic extractor does not take: a SPLINE goes through the exact spline extractor
/// (#1430), anything else is an unmapped source class. The record's own class is the entity's, so
/// a one-span spline that became a Bezier is exact and an unwrapped periodic one is equivalent.
fn classify_spline_leaf(f: &FlatEntity, gate: &Gate) -> Result<Done, PvecConvertError> {
    let class = f.entity.provenance.class_name.as_str();
    let spline = match dwg_spline::extract(&f.entity) {
        Ok(s) => s,
        Err(SplineError::NotSpline) => {
            return Ok(without_geometry(
                crate::pvec::classify_unmapped(&f.entity),
                Vec::new(),
            ));
        }
        Err(e) => {
            return Ok(without_geometry(
                e.classification(class),
                vec![format!("{class}: {e}")],
            ));
        }
    };
    let record = match Normalized2DGeometry::from_spline(&spline) {
        Ok(r) => r,
        Err(e) => {
            return Ok(without_geometry(
                Classification::refused(ReasonCode::DegenerateGeometry, class),
                vec![format!("{class}: {e}")],
            ));
        }
    };
    promote_leaf(f, record, gate)
}

/// A record PVec can carry, checked against the gate: its capability must be allowed, and under
/// an exact-only profile its class must be `SupportedExact`.
fn gated(
    record: Normalized2DGeometry,
    prov: &ProvenanceStub,
    gate: &Gate,
) -> Result<Done, PvecConvertError> {
    let class = prov.class_name.as_str();
    let row = kind_of(&record).map(row_for);
    let capability = row.map(|r| {
        (
            CapabilityId::new(u32::from(r.capability.id)),
            r.capability.name,
        )
    });
    let equivalent = record.fidelity() == FidelityClass::SupportedEquivalent;
    // A curve shape the mapping table has no row for has no capability to check.
    if let Some((id, name)) = capability
        && (!gate.allowed.contains(&id) || (gate.exact_only && equivalent))
    {
        let mut warnings = Vec::new();
        match gate.policy {
            CapabilityPolicy::RejectInput => {
                return Err(PvecConvertError::CapabilityNotAllowed {
                    capability: id,
                    name,
                    source_class: class.to_owned(),
                    handle: prov.handle.map(|h| h.get()),
                    layer: prov.layer.clone(),
                    block_path: prov.block_path.to_string(),
                });
            }
            CapabilityPolicy::Downgrade => {
                warnings.push(format!(
                    "{class}: downgrade-unavailable: no exact weaker representation of {name}"
                ));
            }
            CapabilityPolicy::RejectEntity => {}
        }
        warnings.push(format!("{class}: needs capability {name} ({})", id.get()));
        return Ok(without_geometry(
            Classification::refused(ReasonCode::CapabilityNotAllowed, class),
            warnings,
        ));
    }
    let basis = if equivalent {
        Basis::Equivalent
    } else {
        Basis::Structural
    };
    Ok(Done {
        classification: Classification::supported(basis, class),
        geometry: record,
        warnings: Vec::new(),
    })
}

/// The diagnostic for one entity, or `None` for an entity that became geometry: a supported entity
/// makes none, so a drawing that converts cleanly allocates nothing here.
fn entity_diagnostic(
    item: &Item<'_>,
    raw: &RawEntity,
    c: &Classification,
    layout: &str,
) -> Option<Diagnostic> {
    let code = match c.class() {
        FidelityClass::SupportedExact | FidelityClass::SupportedEquivalent => return None,
        _ if matches!(item, Item::Skipped(_)) => DiagnosticCode::InsertNotExpanded,
        FidelityClass::RequiresPVecCapability | FidelityClass::RequiresExtension => {
            DiagnosticCode::EntityRequiresCapability
        }
        FidelityClass::IgnoredByPolicy => DiagnosticCode::EntityIgnoredByPolicy,
        _ => DiagnosticCode::EntityUnsupported,
    };
    let id = c
        .reason()
        .map(|r| r.code())
        .or_else(|| c.policy_field())
        .unwrap_or("");
    let p = &raw.provenance;
    let mut d = Diagnostic::new(
        code,
        id,
        format!("{} {}: {id}", p.class_name, c.class().tag()),
    );
    d.layout = Some(layout.to_owned());
    d.handle = p.handle.map(|h| h.get());
    d.class = Some(p.class_name.clone());
    d.block_path = Some(p.block_path.to_string());
    d.layer = Some(p.layer.clone());
    Some(d)
}

fn diagnostic_event(d: &Diagnostic) -> EngineEvent {
    EngineEvent::ConversionDiagnostic {
        code: d.code.name().to_owned(),
        severity: format!("{:?}", d.severity).to_lowercase(),
        message_id: d.message_id.clone(),
        handle: d.handle,
        class: d.class.clone(),
        block_path: d.block_path.clone(),
        layer: d.layer.clone(),
    }
}

/// Tells the observer why a run failed, then hands the result on.
fn report_failure<T>(
    observer: Option<&dyn EngineObserver>,
    result: Result<T, PvecConvertError>,
) -> Result<T, PvecConvertError> {
    if let (Err(e), Some(o)) = (&result, observer) {
        o.on_event(diagnostic_event(&e.diagnostic()));
    }
    result
}

impl PvecConvertError {
    /// The diagnostic for this failure: its code, severity and stage, a stable message id, and the
    /// entity's handle, class, layer and block path when the failure is about one entity or one
    /// INSERT chain.
    #[must_use]
    pub fn diagnostic(&self) -> Diagnostic {
        let d = |code, id: &str| Diagnostic::new(code, id, self.to_string());
        match self {
            PvecConvertError::Open(_) => d(DiagnosticCode::SourceOpenFailed, "open"),
            PvecConvertError::View(_) => d(DiagnosticCode::ViewFailed, "view"),
            PvecConvertError::Flatten(e) => {
                let (id, path) = match e {
                    FlattenError::BlockRecursionLimit { block_path, .. } => {
                        ("block-recursion", Some(block_path))
                    }
                    FlattenError::Limit { block_path, .. } => ("limit", Some(block_path)),
                    FlattenError::EntitiesPerBlock { block_path, .. } => {
                        ("entities-per-block", Some(block_path))
                    }
                    FlattenError::Compose(_) => ("compose", None),
                    _ => ("provider", None),
                };
                let mut out = d(DiagnosticCode::BlocksFailed, id);
                out.block_path = path.map(ToString::to_string);
                out
            }
            PvecConvertError::UnitRequired => d(DiagnosticCode::UnitRequired, "unit-required"),
            PvecConvertError::Write(_) | PvecConvertError::Components(_) => {
                d(DiagnosticCode::WriteFailed, "write")
            }
            PvecConvertError::NothingStorable { .. } => {
                d(DiagnosticCode::WriteFailed, "nothing-storable")
            }
            PvecConvertError::Tally(_) | PvecConvertError::Report(_) => {
                d(DiagnosticCode::ReportFailed, "report")
            }
            PvecConvertError::Publish(_) => d(DiagnosticCode::PublishFailed, "publish"),
            PvecConvertError::CapabilityNotAllowed {
                source_class,
                handle,
                layer,
                block_path,
                ..
            } => {
                let mut out = d(DiagnosticCode::CapabilityRejected, "capability-not-allowed");
                out.class = Some(source_class.clone());
                out.handle = *handle;
                out.layer = Some(layer.clone());
                out.block_path = Some(block_path.clone());
                out
            }
            PvecConvertError::Cancelled => d(DiagnosticCode::Cancelled, "cancelled"),
        }
    }
}
