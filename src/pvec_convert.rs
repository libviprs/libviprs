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

mod capability;
mod georef;
#[cfg(feature = "pvec-raster")]
mod pdf;
pub use capability::{CapabilityPolicy, ConversionProfile};
pub use georef::{
    GeoreferenceError, GeoreferenceHandoff, HandoffV1, SourceBox, UnitRecord, georef_path,
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
use crate::pvec::provenance::NoProvenance;
use crate::pvec::publish::{PublishError, StreamedPublishError, publish, publish_streamed};
use crate::pvec::regions::{RegionConfig, RoomLabel, WallEdge, derive_regions, semantic_stream};
use crate::pvec::report::{
    BoundsSplit, CapabilitySummary, ComponentsSummary, ConversionReport, EntityRow, RegionsSummary,
    ReportBuilder, ReportError, UnresolvedXrefRow, ViewSummary, XrefSummary,
};
use crate::pvec::style::{LineStyleComponent, StyleReport, dwg_line_style, dwg_style_input};
use crate::pvec::transform::{ComposedTransform, promote};
use crate::pvec::writer::{
    CapabilityId, LeafConfig, OpenArtifact, SpillConfig, WriteConfig, WriteError, open_geometry,
};
use crate::pvec::{
    DwgProvider, EntityId, EntityScope, EntityStyle, ExcludedEntity, FidelityClass, OrderingPolicy,
    ProvenanceStub, ProviderError, RawEntity, RawGeometry, SourceLimits, SourceUnit, ViewProfile,
    ViewProfileError, canonical_cmp, order_entities, select_layout,
};
use crate::pvec::{Normalized2DGeometry, ViewedEntity};
use crate::sync_queue::bounded;

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
    /// The georeference sidecar could not be built or written.
    #[error("the georeference sidecar could not be produced")]
    Georeference(#[source] GeoreferenceError),
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
    calibration: Option<String>,
    control_points: Vec<[f64; 2]>,
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
            control_points: Vec::new(),
            observer: None,
            queue_capacity: DEFAULT_QUEUE_CAPACITY,
            components: StreamSelection::NONE,
            regions: None,
        }
    }

    /// The component streams to write beside the geometry: layer, line style and provenance, each
    /// on its own (#1525). Off by default, and the geometry is the same bytes either way.
    #[must_use]
    pub fn components(mut self, select: StreamSelection) -> Self {
        self.components = select;
        self
    }

    /// Derive rooms and corridors from the wall linework and write them as PVec's semantic class
    /// stream (#1526). Off (`None`) by default. The walls are the straight lines the default
    /// [`ElementRules`] call a wall, the room names come from TEXT, and the geometry is the same
    /// bytes either way: the segments that close a door gap are never written as geometry, so a room
    /// that runs along one is left out of the stream and counted in the report.
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

    /// The drawing's unit, overriding what the provider reads.
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

    /// The identifier of the calibration record the application holds for this drawing. It goes
    /// into the georeference sidecar and nowhere else: the artifact's bytes do not depend on it.
    #[must_use]
    pub fn calibration(mut self, id: impl Into<String>) -> Self {
        self.calibration = Some(id.into());
        self
    }

    /// Control points in source space (drawing coordinates, no world coordinates), for the
    /// georeference sidecar.
    #[must_use]
    pub fn control_points(mut self, points: impl IntoIterator<Item = [f64; 2]>) -> Self {
        self.control_points = points.into_iter().collect();
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
        let c = self.classify(&provider)?;
        c.tally
            .finish(c.encountered)
            .map_err(PvecConvertError::Tally)
    }

    /// As [`probe`](Self::probe), with provider `P`.
    ///
    /// # Errors
    ///
    /// As [`probe`](Self::probe).
    pub fn probe_with<P: DwgProvider>(self) -> Result<FidelityReport, PvecConvertError> {
        let provider = P::open(self.source).map_err(PvecConvertError::Open)?;
        let c = self.classify(&provider)?;
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
                    let declared = provider.source_unit();
                    let unit = self.unit.unwrap_or(declared);
                    self.run(&provider, declared, unit)
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
                    // The trait reports no unit, so the drawing's own is unknown; the caller's is
                    // applied.
                    self.run(&provider, SourceUnit::Unknown, unit)
                })
        };
        report_failure(observer.as_deref(), result)
    }

    /// Everything a conversion decides about its entities, up to but not including the write.
    /// [`probe`](Self::probe) stops here, so it and [`convert`](Self::convert) cannot disagree.
    fn classify<P: DwgProvider>(&self, provider: &P) -> Result<Classified, PvecConvertError> {
        let Self {
            source,
            view_profile,
            cancel,
            gate,
            observer,
            queue_capacity,
            components,
            regions,
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
        let excluded_inserts: BTreeSet<u64> = view
            .excluded
            .iter()
            .filter_map(|x| x.handle.map(|h| h.get()))
            .collect();

        let mut items: Vec<Item<'_>> = Vec::new();
        let mut raws: Vec<RawEntity> = Vec::new();
        for f in &flat.entities {
            let p = &f.entity.provenance;
            if kept.contains(&(
                p.handle.map(|h| h.get()),
                p.block_path.to_string(),
                p.class_name.clone(),
            )) {
                items.push(Item::Leaf(f));
                raws.push(f.entity.clone());
            }
        }
        for x in &view.excluded {
            items.push(Item::Excluded(x));
            raws.push(stub(
                x.handle.map(|h| h.get()),
                &x.class_name,
                x.block_path.clone(),
                &x.layer,
                x.reason.code(),
            ));
        }
        for s in &flat.skipped {
            if s.insert_handle
                .is_some_and(|h| excluded_inserts.contains(&h.get()))
            {
                continue;
            }
            items.push(Item::Skipped(s));
            raws.push(stub(
                s.insert_handle.map(|h| h.get()),
                "INSERT",
                s.block_path.clone(),
                "",
                s.reason.code(),
            ));
        }
        let encountered = raws.len() as u64;

        // Ids are positions in the canonical order. `order_entities` hands the ids back without
        // the permutation, so the same stable sort is run here to pair each id with its item.
        let mut perm: Vec<usize> = (0..raws.len()).collect();
        perm.sort_by(|&a, &b| canonical_cmp(a as u64, &raws[a], b as u64, &raws[b]));
        let ordered = order_entities(provider.name(), "", &OrderingPolicy::new(), raws);

        // The classifying side produces, a consumer thread turns each outcome into a row, a
        // record, a diagnostic and a tally entry. They meet in a bounded queue, so a slow consumer
        // makes the producer wait instead of letting outcomes pile up.
        let mut style_report = StyleReport::default();
        let mut style_bad = 0u64;
        // The wall lines and the room names the regions step reads, in canonical order.
        let rules = regions.is_some().then(ElementRules::default_profile);
        let mut walls: Vec<WallEdge> = Vec::new();
        let mut labels: Vec<RoomLabel> = Vec::new();
        let (tx, rx) = bounded::<Outcome>(*queue_capacity);
        let sent = AtomicU64::new(0);
        let consumed = std::thread::scope(|scope| {
            let consumer = scope.spawn(|| consume(rx, &sent, observer));
            let mut failure = None;
            for (n, &at) in perm.iter().enumerate() {
                if n.is_multiple_of(BATCH) && cancel.is_cancelled() {
                    failure = Some(PvecConvertError::Cancelled);
                    break;
                }
                let raw = &ordered.entities[n].entity;
                let done = match classify_item(&items[at], raw, gate) {
                    Ok(done) => done,
                    Err(e) => {
                        failure = Some(e);
                        break;
                    }
                };
                let mut diagnostic =
                    entity_diagnostic(&items[at], raw, &done.classification, &layout.name);
                if let Some(d) = &mut diagnostic {
                    d.source = Some(source_name.clone());
                }
                let id = ordered.entities[n].id;
                if let (Some(rules), Item::Leaf(f)) = (&rules, &items[at]) {
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
                            if matches!(raw.provenance.class_name.as_str(), "TEXT" | "MTEXT") =>
                        {
                            let anchor =
                                raw.provenance.handle.and_then(|h| provider.text_anchor(h));
                            if let Some((text, at)) = anchor.filter(|(t, _)| !t.trim().is_empty()) {
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
                    let style = components.line_style.then(|| {
                        let input = dwg_style_input(&raw.style, true);
                        style_report.count_colour(
                            &raw.provenance.class_name,
                            u64::from(input.colour_index.is_some()),
                        );
                        let c = dwg_line_style(id, &input).ok();
                        style_bad += u64::from(c.is_none());
                        c
                    });
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
                        style: style.flatten(),
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
                    break;
                }
                sent.fetch_add(1, Ordering::Release);
            }
            drop(tx);
            let consumed = consumer.join().expect("the consumer thread does not panic");
            failure.map_or(Ok(consumed), Err)
        })?;
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
            style_report,
            style_bad,
            walls,
            labels,
        })
    }

    fn run<P: DwgProvider>(
        self,
        provider: &P,
        declared: SourceUnit,
        unit: SourceUnit,
    ) -> Result<PvecConversionOutput, PvecConvertError> {
        georef::check_finite(&self.control_points).map_err(PvecConvertError::Georeference)?;
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
            style_report,
            style_bad,
            walls,
            labels,
        } = self.classify(provider)?;
        let Self {
            dest,
            leaves,
            spill,
            cancel,
            gate,
            calibration,
            control_points,
            observer,
            components: select,
            regions,
            ..
        } = self;
        let tell = |event: EngineEvent| {
            if let Some(o) = &observer {
                o.on_event(event);
            }
        };

        let sidecar = georef_path(dest);
        // The sidecar is staged first and moved into place once the pair is published, so a failed
        // or cancelled run leaves the previous one as it was.
        let staged = PathBuf::from(format!("{}.partial", sidecar.display()));
        let write_error = |e: WriteError| match e {
            WriteError::Cancelled => PvecConvertError::Cancelled,
            other => PvecConvertError::Write(other),
        };
        // What both paths do once the artifact is written: tell the observer, close the report against
        // what went in, and stage the georeference sidecar. It runs once, so it takes the rows.
        // The tally is sealed here and not at classification: an entity the classifier called
        // supported can still be one PVec's container refuses, and it is counted as that.
        let seal =
            |bounds: &BoundsReport,
             written: u64,
             length: u64,
             refused: &[(EntityId, String)],
             components: Option<ComponentsSummary>,
             found: Option<RegionsSummary>|
             -> Result<(ConversionReport, FidelityReport, String), PvecConvertError> {
                for (id, rule) in refused {
                    let from = &stored
                        .iter()
                        .find(|(s, _)| s == id)
                        .expect("a refused entity was written from a stored classification")
                        .1;
                    let to =
                        Classification::refused(ReasonCode::PvecCannotStore, from.source_class());
                    tally.reclassify(from, &to);
                    rows.refuse(*id, &to, format!("PVec's writer refused it ({rule})"));
                }
                let fidelity = tally.finish(encountered).map_err(PvecConvertError::Tally)?;
                tell(EngineEvent::ConversionArtifactWritten {
                    entities: written,
                    bytes: length,
                });
                let mut report = rows
                    .finish(
                        encountered,
                        view,
                        xrefs,
                        BoundsSplit {
                            exact: bounds.exact_count() as u64,
                            hull_upper_bound: bounds.hull_count() as u64,
                        },
                    )
                    .map_err(PvecConvertError::Report)?;
                report.capabilities = Some(gate.summary());
                report.diagnostics = diagnostics;
                report.components = components;
                report.regions = found;
                report.georeference = sidecar
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned());
                let json = report.to_json_string().map_err(PvecConvertError::Report)?;
                GeoreferenceHandoff::V1(HandoffV1 {
                    bounds: bounds.drawing_bounds().map(|b| SourceBox {
                        min: [b.min().x, b.min().y],
                        max: [b.max().x, b.max().y],
                    }),
                    unit: UnitRecord::new(declared, unit),
                    calibration_id: calibration,
                    control_points,
                })
                .write_to(&staged)
                .map_err(PvecConvertError::Georeference)?;
                Ok((report, fidelity, json))
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
        let items = || geometries.iter().map(|(id, g)| (*id, g));
        // The geometry is written first, then the streams against the ids the writer kept, because a
        // refused curve moves every id after it. The streams go in after the geometry, so they cannot
        // change a geometry block.
        let open = |spill: Option<&SpillConfig>,
                    on_chunk: &mut dyn FnMut(u64, u64)|
         -> Result<
            (
                OpenArtifact,
                Option<ComponentsSummary>,
                Option<RegionsSummary>,
            ),
            PvecConvertError,
        > {
            let mut open =
                open_geometry(&config, items(), Vec::new(), None, spill, &cancel, on_chunk)
                    .map_err(write_error)?;
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
                let mut made = ComponentsSummary {
                    streams: built.names.iter().map(|n| (*n).to_owned()).collect(),
                    dropped: BTreeMap::new(),
                    colours_ignored: style_report.colour_total(),
                };
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
        let published = match &spill {
            None => open(None, &mut on_chunk).and_then(|(open, summary, found)| {
                let artifact = open.finish().map_err(write_error)?;
                let (report, fidelity, json) = seal(
                    &artifact.bounds,
                    artifact.written,
                    artifact.bytes.len() as u64,
                    &artifact.refused,
                    summary,
                    found,
                )?;
                publish(dest, &artifact.bytes, json.as_bytes(), &cancel, &mut |_| {}).map_err(
                    |e| match e {
                        PublishError::Cancelled => PvecConvertError::Cancelled,
                        other => PvecConvertError::Publish(other),
                    },
                )?;
                Ok((report, fidelity, artifact.written))
            }),
            Some(spill) => {
                let mut sealed = None;
                publish_streamed(dest, &cancel, &mut |_| {}, |out| {
                    let (open, summary, found) = open(Some(spill), &mut on_chunk)?;
                    let artifact = open
                        .seal_into(&cancel, &mut on_chunk, out)
                        .map_err(write_error)?;
                    tell(EngineEvent::ConversionSpoolSpilled {
                        bytes: artifact.length,
                    });
                    let (report, fidelity, json) = seal(
                        &artifact.bounds,
                        artifact.written,
                        artifact.length,
                        &artifact.refused,
                        summary,
                        found,
                    )?;
                    sealed = Some((report, fidelity, artifact.written));
                    Ok(json.into_bytes())
                })
                .map_err(|e| match e {
                    StreamedPublishError::Produce(e) => e,
                    StreamedPublishError::Publish(PublishError::Cancelled) => {
                        PvecConvertError::Cancelled
                    }
                    StreamedPublishError::Publish(other) => PvecConvertError::Publish(other),
                })
                .map(|()| sealed.expect("the producer sealed the report before the commit"))
            }
        }
        .and_then(|done| {
            std::fs::rename(&staged, &sidecar)
                .map_err(|e| PvecConvertError::Georeference(GeoreferenceError::Io(e)))?;
            Ok(done)
        });
        if published.is_err() {
            let _ = std::fs::remove_file(&staged);
        }
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
    style_report: StyleReport,
    /// Entities whose line style the DWG facts could not make into a component.
    style_bad: u64,
    /// The straight wall leaves and the text anchors the regions step reads; empty when it is off.
    walls: Vec<WallEdge>,
    labels: Vec<RoomLabel>,
}

/// What the component streams read off one written entity: where it came from and its line style.
struct Side {
    stub: ProvenanceStub,
    style: Option<LineStyleComponent>,
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
            PvecConvertError::Tally(_) | PvecConvertError::Report(_) => {
                d(DiagnosticCode::ReportFailed, "report")
            }
            PvecConvertError::Publish(_) => d(DiagnosticCode::PublishFailed, "publish"),
            PvecConvertError::Georeference(_) => {
                d(DiagnosticCode::GeoreferenceFailed, "georeference")
            }
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
