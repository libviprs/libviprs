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

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use thiserror::Error;

mod capability;
pub use capability::{CapabilityPolicy, ConversionProfile};

use crate::CancelToken;
use crate::cad::CadSource;
use crate::pvec::dwg_acadrust::AcadrustProvider;
use crate::pvec::dwg_basic::{self, BasicError};
use crate::pvec::fidelity::{Basis, Classification, FidelityReport, ReasonCode, Tally, TallyError};
use crate::pvec::flatten::{
    BlockResolver, FlatEntity, FlattenBounds, FlattenError, LineageStore, SkipReason, SkippedInsert,
};
use crate::pvec::mapping::{kind_of, row_for};
use crate::pvec::provenance::NoProvenance;
use crate::pvec::publish::{PublishError, publish};
use crate::pvec::report::{
    BoundsSplit, CapabilitySummary, ConversionReport, EntityRow, ReportBuilder, ReportError,
    UnresolvedXrefRow, ViewSummary, XrefSummary,
};
use crate::pvec::transform::{ComposedTransform, promote};
use crate::pvec::writer::{CapabilityId, LeafConfig, WriteConfig, WriteError, write_geometry};
use crate::pvec::{
    DwgProvider, EntityId, EntityScope, EntityStyle, ExcludedEntity, FidelityClass, OrderingPolicy,
    ProvenanceStub, ProviderError, RawEntity, RawGeometry, SourceLimits, SourceUnit, ViewProfile,
    ViewProfileError, canonical_cmp, order_entities, select_layout,
};
use crate::pvec::{Normalized2DGeometry, ViewedEntity};

/// How often the conversion looks at the cancel token while it classifies entities.
const CANCEL_EVERY: usize = 1024;

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
}

/// One conversion, configured the way `EngineBuilder` configures a pyramid run.
pub struct PvecConversion<'a> {
    source: CadSource<'a>,
    dest: &'a Path,
    view_profile: ViewProfile,
    unit: Option<SourceUnit>,
    leaves: Option<LeafConfig>,
    cancel: CancelToken,
    gate: Gate,
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
            cancel: CancelToken::new(),
            gate: Gate::of(ConversionProfile::Default),
        }
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

    /// The cancel token, polled before any file exists and while entities are classified.
    #[must_use]
    pub fn cancel(mut self, cancel: CancelToken) -> Self {
        self.cancel = cancel;
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
    /// [`convert`](Self::convert) would report. Under [`CapabilityPolicy::RejectInput`] it fails
    /// the way the conversion would.
    ///
    /// # Errors
    ///
    /// As [`convert`](Self::convert), except that no unit is needed and nothing is written.
    pub fn probe(self) -> Result<FidelityReport, PvecConvertError> {
        let provider = AcadrustProvider::open(self.source).map_err(PvecConvertError::Open)?;
        Ok(self.classify(&provider)?.fidelity)
    }

    /// As [`probe`](Self::probe), with provider `P`.
    ///
    /// # Errors
    ///
    /// As [`probe`](Self::probe).
    pub fn probe_with<P: DwgProvider>(self) -> Result<FidelityReport, PvecConvertError> {
        let provider = P::open(self.source).map_err(PvecConvertError::Open)?;
        Ok(self.classify(&provider)?.fidelity)
    }

    /// Converts with the default provider, acadrust. The unit is the header's unless
    /// [`unit`](Self::unit) was set.
    ///
    /// # Errors
    ///
    /// [`PvecConvertError`]. After any error the destination and its report are as they were.
    pub fn convert(self) -> Result<PvecConversionOutput, PvecConvertError> {
        if self.cancel.is_cancelled() {
            return Err(PvecConvertError::Cancelled);
        }
        let provider = AcadrustProvider::open(self.source).map_err(PvecConvertError::Open)?;
        let unit = self.unit.unwrap_or_else(|| provider.source_unit());
        self.run(&provider, unit)
    }

    /// Converts with provider `P`. The trait carries no unit, so [`unit`](Self::unit) is required.
    ///
    /// # Errors
    ///
    /// [`PvecConvertError::UnitRequired`] without a unit, otherwise as [`convert`](Self::convert).
    pub fn convert_with<P: DwgProvider>(self) -> Result<PvecConversionOutput, PvecConvertError> {
        if self.cancel.is_cancelled() {
            return Err(PvecConvertError::Cancelled);
        }
        let unit = self.unit.ok_or(PvecConvertError::UnitRequired)?;
        let provider = P::open(self.source).map_err(PvecConvertError::Open)?;
        self.run(&provider, unit)
    }

    /// Everything a conversion decides about its entities, up to but not including the write.
    /// [`probe`](Self::probe) stops here, so it and [`convert`](Self::convert) cannot disagree.
    fn classify<P: DwgProvider>(&self, provider: &P) -> Result<Classified, PvecConvertError> {
        let Self {
            source,
            view_profile,
            cancel,
            gate,
            ..
        } = self;
        let view = view_profile
            .apply(provider)
            .map_err(PvecConvertError::View)?;
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

        let mut tally = Tally::new();
        let mut rows = ReportBuilder::new();
        let mut geometries: Vec<(EntityId, Normalized2DGeometry)> = Vec::new();
        for (n, &at) in perm.iter().enumerate() {
            if n % CANCEL_EVERY == 0 && cancel.is_cancelled() {
                return Err(PvecConvertError::Cancelled);
            }
            let id = ordered.entities[n].id;
            let raw = &ordered.entities[n].entity;
            let done = classify_item(&items[at], raw, gate)?;
            tally.record(&done.classification);
            rows.push(EntityRow::new(
                id,
                &done.classification,
                raw.provenance.handle.map(|h| h.get()),
                done.geometry
                    .geometry()
                    .and_then(|_| kind_of(&done.geometry))
                    .map(|k| row_for(k).pvec_type),
                done.warnings,
            ));
            geometries.push((id, done.geometry));
        }
        let fidelity = tally.finish(encountered).map_err(PvecConvertError::Tally)?;
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
            geometries,
            rows,
            fidelity,
            encountered,
            view: ViewSummary::from_report(&view.report),
            xrefs,
        })
    }

    fn run<P: DwgProvider>(
        self,
        provider: &P,
        unit: SourceUnit,
    ) -> Result<PvecConversionOutput, PvecConvertError> {
        let Classified {
            geometries,
            rows,
            fidelity,
            encountered,
            view,
            xrefs,
        } = self.classify(provider)?;
        let Self {
            dest,
            leaves,
            cancel,
            gate,
            ..
        } = self;

        let artifact = write_geometry(
            &WriteConfig { unit, leaves },
            geometries.iter().map(|(id, g)| (*id, g)),
            Vec::new(),
            &cancel,
        )
        .map_err(|e| match e {
            WriteError::Cancelled => PvecConvertError::Cancelled,
            other => PvecConvertError::Write(other),
        })?;

        let mut report = rows
            .finish(
                encountered,
                view,
                xrefs,
                BoundsSplit {
                    exact: artifact.bounds.exact_count() as u64,
                    hull_upper_bound: artifact.bounds.hull_count() as u64,
                },
            )
            .map_err(PvecConvertError::Report)?;
        report.capabilities = Some(gate.summary());
        let json = report.to_json_string().map_err(PvecConvertError::Report)?;
        publish(dest, &artifact.bytes, json.as_bytes(), &cancel, &mut |_| {}).map_err(
            |e| match e {
                PublishError::Cancelled => PvecConvertError::Cancelled,
                other => PvecConvertError::Publish(other),
            },
        )?;
        Ok(PvecConversionOutput {
            artifact: dest.to_path_buf(),
            report,
            fidelity,
            written: artifact.written,
        })
    }
}

/// What `classify` hands on: one record and one row per entity, and the counts that check them.
struct Classified {
    geometries: Vec<(EntityId, Normalized2DGeometry)>,
    rows: ReportBuilder,
    fidelity: FidelityReport,
    encountered: u64,
    view: ViewSummary,
    xrefs: XrefSummary,
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
        Err(BasicError::NotBasic) => {
            return Ok(without_geometry(
                crate::pvec::classify_unmapped(&f.entity),
                Vec::new(),
            ));
        }
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
    // A leaf a layout owns directly has the identity map. A nested one is pushed through the map
    // flattening composed for it, as one level named by its innermost INSERT.
    let path = &f.entity.provenance.block_path;
    let handle = f.entity.provenance.handle.map(|h| h.get());
    let Some(step) = path.steps().last().cloned() else {
        return gated(record, class, handle, gate);
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
        Ok(p) if p.record().geometry().is_some() => gated(p.record().clone(), class, handle, gate),
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

/// A record PVec can carry, checked against the gate: its capability must be allowed, and under
/// an exact-only profile its class must be `SupportedExact`.
fn gated(
    record: Normalized2DGeometry,
    class: &str,
    handle: Option<u64>,
    gate: &Gate,
) -> Result<Done, PvecConvertError> {
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
                    handle,
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
