//! The provider differential harness (issue #1395, N2.7 and N4.8).
//!
//! A second DWG provider is only replaceable if it can be held to the first one's answers. This
//! module runs any number of [`DwgProvider`]s over the same fixtures and compares, per entity,
//! what each one says:
//!
//! * the structural parameters the entity carries (and the composed insert map),
//! * the N3.5 bucket, per entity and not in aggregate, with its reason code,
//! * the PVec semantic type of N6.2's table.
//!
//! Nothing here is a trait of its own: the harness is generic over [`DwgProvider`], so a second
//! backend is one more call to [`Matrix::run`].
//!
//! # What a pass means
//!
//! Two providers agreeing is [`AGREEMENT_CLASS`], `DUAL_ORACLE_EQUIVALENT`, and never source
//! truth: both read the same bytes. The fixture's own evidence class (`KNOWN_TRUTH_EXACT` for the
//! known-truth set) is EPIC O's, comes in with the fixture and is printed beside the agreement,
//! never replaced by it. A run with fewer than two providers is [`Verdict::Partial`], which is not
//! a pass, whatever the single provider said.
//!
//! # Allowed differences
//!
//! [`Deltas`] is `tests/fixtures/pvec/provider_deltas.toml`: one entry per fixture, entity and
//! field, each with a written reason. A difference that is not listed fails. An entry that no
//! longer matches a difference fails too, so the list cannot rot.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::PathBuf;

use thiserror::Error;

use crate::cad::CadSource;

use super::dwg_basic::{self, BasicError, BasicKind};
use super::fidelity::{Classification, classify_unmapped};
use super::flatten::{BlockResolver, FlatEntity, FlattenBounds, LineageStore};
use super::mapping::{NORMALIZED_TO_PVEC, NormalizedKind};
use super::provenance::NoProvenance;
use super::provider::{
    DwgProvider, EntityScope, LayoutKind, ProviderError, RawEntity, RawGeometry,
};

/// What two providers agreeing is called. Never source truth.
pub const AGREEMENT_CLASS: &str = "DUAL_ORACLE_EQUIVALENT";

/// One fixture to run.
#[derive(Debug, Clone)]
pub struct Fixture {
    /// The fixture's id, used in reports and in the deltas file.
    pub id: String,
    /// The drawing.
    pub path: PathBuf,
    /// The fixture's own evidence class, as its corpus states it.
    pub evidence: String,
}

/// What one provider said about one flattened entity.
#[derive(Debug, Clone, PartialEq)]
pub struct EntityRow {
    /// `<instance path>#<handle>`, so a leaf reached through two INSERTs has two rows.
    pub key: String,
    /// The drawing's class name for it.
    pub class_name: String,
    /// Its own layer.
    pub layer: String,
    /// The structural parameters, by name, and the composed map as `matrix.*`.
    pub params: Vec<(String, f64)>,
    /// The N3.5 class tag.
    pub bucket: String,
    /// The reason code, empty when there is none.
    pub reason: String,
    /// The PVec semantic type of N6.2, empty when there is none.
    pub pvec_type: String,
}

/// A difference between two providers.
#[derive(Debug, Clone, PartialEq)]
pub struct Difference {
    /// The fixture.
    pub fixture: String,
    /// The entity key.
    pub entity: String,
    /// `present`, `class_name`, `layer`, `bucket`, `reason`, `pvec_type` or `param:<name>`.
    pub field: String,
    /// The first provider's name.
    pub left_provider: String,
    /// What it said.
    pub left: String,
    /// The other provider's name.
    pub right_provider: String,
    /// What it said.
    pub right: String,
}

/// An allowed difference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllowedDelta {
    /// The fixture.
    pub fixture: String,
    /// The entity key.
    pub entity: String,
    /// The field.
    pub field: String,
    /// Why the providers differ here.
    pub reason: String,
}

/// The allowed-difference list.
#[derive(Debug, Clone, Default)]
pub struct Deltas {
    entries: Vec<AllowedDelta>,
}

/// Why a deltas file or a run was refused.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DiffError {
    /// The deltas file is not valid.
    #[error("provider_deltas.toml: {0}")]
    Deltas(String),
    /// A provider failed on a fixture.
    #[error("{provider} on {fixture}: {source}")]
    Provider {
        /// Provider name.
        provider: String,
        /// Fixture id.
        fixture: String,
        /// Why.
        source: ProviderError,
    },
    /// The flattener refused a fixture.
    #[error("{provider} on {fixture}: {detail}")]
    Flatten {
        /// Provider name.
        provider: String,
        /// Fixture id.
        fixture: String,
        /// The flattener's message.
        detail: String,
    },
}

impl Deltas {
    /// Parses the file.
    ///
    /// # Errors
    ///
    /// [`DiffError::Deltas`] for an entry with no reason, a blank reason, an unknown key or a
    /// repeated (fixture, entity, field).
    pub fn parse(text: &str) -> Result<Self, DiffError> {
        let bad = |line: usize, what: &str| DiffError::Deltas(format!("line {line}: {what}"));
        let mut entries: Vec<AllowedDelta> = Vec::new();
        let mut open: Option<BTreeMap<String, String>> = None;
        let close = |open: &mut Option<BTreeMap<String, String>>,
                     entries: &mut Vec<AllowedDelta>,
                     line: usize|
         -> Result<(), DiffError> {
            let Some(mut m) = open.take() else {
                return Ok(());
            };
            let mut take = |k: &str| {
                m.remove(k)
                    .ok_or_else(|| bad(line, &format!("an entry has no `{k}`")))
            };
            let (fixture, entity, field) = (take("fixture")?, take("entity")?, take("field")?);
            let reason = take("reason")?;
            if reason.trim().is_empty() {
                return Err(bad(line, "an entry needs a written reason"));
            }
            if entries
                .iter()
                .any(|e| (&e.fixture, &e.entity, &e.field) == (&fixture, &entity, &field))
            {
                return Err(bad(
                    line,
                    "the same fixture, entity and field is listed twice",
                ));
            }
            entries.push(AllowedDelta {
                fixture,
                entity,
                field,
                reason,
            });
            Ok(())
        };
        let mut last = 0;
        for (i, raw) in text.lines().enumerate() {
            last = i + 1;
            let l = raw.trim();
            if l.is_empty() || l.starts_with('#') {
                continue;
            }
            if l == "[[delta]]" {
                close(&mut open, &mut entries, last)?;
                open = Some(BTreeMap::new());
                continue;
            }
            let Some(m) = open.as_mut() else {
                return Err(bad(last, "expected `[[delta]]`"));
            };
            let (k, v) = l
                .split_once('=')
                .ok_or_else(|| bad(last, "expected `key = \"value\"`"))?;
            let (k, v) = (k.trim(), v.trim());
            if !["fixture", "entity", "field", "reason"].contains(&k) {
                return Err(bad(last, &format!("unknown key `{k}`")));
            }
            let inner = v
                .strip_prefix('"')
                .and_then(|v| v.strip_suffix('"'))
                .ok_or_else(|| bad(last, "a value is a double-quoted string"))?;
            let value = inner.replace("\\\"", "\"").replace("\\\\", "\\");
            if m.insert(k.to_string(), value).is_some() {
                return Err(bad(last, &format!("`{k}` is set twice")));
            }
        }
        close(&mut open, &mut entries, last)?;
        Ok(Self { entries })
    }

    /// The entries.
    #[must_use]
    pub fn entries(&self) -> &[AllowedDelta] {
        &self.entries
    }
}

/// The outcome of a comparison.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum Verdict {
    /// Two or more providers ran, every difference is listed and every listed entry happened.
    Equivalent,
    /// Fewer than two providers ran: no differential was possible. Not a pass.
    Partial {
        /// The providers that did not run.
        missing: Vec<String>,
    },
    /// A difference is not listed, or a listed one did not happen.
    Failed {
        /// Differences with no entry.
        unlisted: Vec<Difference>,
        /// Entries that matched nothing.
        stale: Vec<AllowedDelta>,
    },
}

impl Verdict {
    /// Whether this is a pass. A partial run is not.
    #[must_use]
    pub fn passed(&self) -> bool {
        matches!(self, Verdict::Equivalent)
    }
}

/// What one provider produced for one fixture.
#[derive(Debug, Clone)]
struct FixtureRun {
    fixture: String,
    evidence: String,
    rows: Vec<EntityRow>,
}

#[derive(Debug, Clone)]
struct ProviderRun {
    provider: String,
    fixtures: Vec<FixtureRun>,
}

/// Providers run over fixtures, and the ones that could not be.
#[derive(Debug, Clone)]
pub struct Matrix {
    corpus: String,
    runs: Vec<ProviderRun>,
    absent: Vec<(String, String)>,
}

fn platform() -> String {
    format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS)
}

fn pvec_type_of(kind: NormalizedKind) -> String {
    NORMALIZED_TO_PVEC
        .iter()
        .find(|r| r.kind == kind)
        .map_or_else(String::new, |r| r.pvec_type.to_string())
}

fn normalized_kind(b: &dwg_basic::ExactBasic) -> NormalizedKind {
    match b.kind {
        BasicKind::Line => NormalizedKind::Line,
        BasicKind::Polyline => {
            if b.segments
                .iter()
                .any(|s| !matches!(s, dwg_basic::SegmentKind::Straight))
            {
                NormalizedKind::Composite
            } else {
                NormalizedKind::Polyline
            }
        }
        BasicKind::Circle => NormalizedKind::Circle,
        BasicKind::Arc => NormalizedKind::CircularArc,
        BasicKind::Ellipse => NormalizedKind::Ellipse,
        BasicKind::EllipticalArc => NormalizedKind::EllipticalArc,
    }
}

fn push3(out: &mut Vec<(String, f64)>, name: &str, p: [f64; 3]) {
    for (axis, v) in ["x", "y", "z"].into_iter().zip(p) {
        out.push((format!("{name}.{axis}"), v));
    }
}

/// The numbers the entity carries, by name, as the provider reported them.
fn params_of(g: &RawGeometry) -> Vec<(String, f64)> {
    let mut o = Vec::new();
    match g {
        RawGeometry::Line { start, end } => {
            push3(&mut o, "start", *start);
            push3(&mut o, "end", *end);
        }
        RawGeometry::Polyline {
            vertices,
            closed,
            elevation,
            normal,
        } => {
            for (i, v) in vertices.iter().enumerate() {
                o.push((format!("v{i}.x"), v.x));
                o.push((format!("v{i}.y"), v.y));
                o.push((format!("v{i}.bulge"), v.bulge));
            }
            o.push(("closed".into(), f64::from(u8::from(*closed))));
            o.push(("elevation".into(), *elevation));
            push3(&mut o, "normal", *normal);
        }
        RawGeometry::Arc {
            center,
            radius,
            start_angle,
            end_angle,
            normal,
        } => {
            push3(&mut o, "center", *center);
            o.push(("radius".into(), *radius));
            o.push(("start_angle".into(), *start_angle));
            o.push(("end_angle".into(), *end_angle));
            push3(&mut o, "normal", *normal);
        }
        RawGeometry::Circle {
            center,
            radius,
            normal,
        } => {
            push3(&mut o, "center", *center);
            o.push(("radius".into(), *radius));
            push3(&mut o, "normal", *normal);
        }
        RawGeometry::Ellipse {
            center,
            major_axis,
            minor_ratio,
            start_parameter,
            end_parameter,
            normal,
        } => {
            push3(&mut o, "center", *center);
            push3(&mut o, "major_axis", *major_axis);
            o.push(("minor_ratio".into(), *minor_ratio));
            o.push(("start_parameter".into(), *start_parameter));
            o.push(("end_parameter".into(), *end_parameter));
            push3(&mut o, "normal", *normal);
        }
        RawGeometry::Spline {
            degree,
            control_points,
            closed,
            ..
        } => {
            o.push(("degree".into(), f64::from(*degree)));
            o.push(("closed".into(), f64::from(u8::from(*closed))));
            for (i, p) in control_points.iter().enumerate() {
                push3(&mut o, &format!("cp{i}"), *p);
            }
        }
        _ => {}
    }
    o
}

fn classify(e: &RawEntity) -> (Classification, String) {
    match dwg_basic::extract(e) {
        Ok(b) => (b.classification.clone(), pvec_type_of(normalized_kind(&b))),
        Err(BasicError::NotBasic) => match e.geometry {
            RawGeometry::Unmapped { .. } => (classify_unmapped(e), String::new()),
            // Splines and INSERTs are not this harness's to classify; they are compared by
            // parameters and tagged so the report says so.
            _ => (
                Classification::supported(
                    super::fidelity::Basis::Structural,
                    e.provenance.class_name.as_str(),
                ),
                String::new(),
            ),
        },
        Err(err) => (
            err.classification(e.provenance.class_name.as_str()),
            String::new(),
        ),
    }
}

fn row_of(f: &FlatEntity, key: String) -> EntityRow {
    let e = &f.entity;
    let (c, pvec_type) = classify(e);
    let mut params = params_of(&e.geometry);
    for (n, v) in ["m00", "m01", "m10", "m11", "tx", "ty"]
        .into_iter()
        .zip(f.matrix())
    {
        params.push((format!("matrix.{n}"), v));
    }
    let not_classified = matches!(e.geometry, RawGeometry::Spline { .. });
    EntityRow {
        key,
        class_name: e.provenance.class_name.clone(),
        layer: e.provenance.layer.clone(),
        params,
        bucket: if not_classified {
            "NotClassified".to_string()
        } else {
            c.class().tag().to_string()
        },
        reason: c
            .reason()
            .map_or_else(String::new, |r| r.code().to_string()),
        pvec_type,
    }
}

impl Matrix {
    /// An empty matrix over the corpus revision `corpus`.
    #[must_use]
    pub fn new(corpus: impl Into<String>) -> Self {
        Self {
            corpus: corpus.into(),
            runs: Vec::new(),
            absent: Vec::new(),
        }
    }

    /// Runs provider `P` over `fixtures`: each fixture's model space, flattened through the
    /// provider, one row per leaf and one per INSERT that could not be expanded.
    ///
    /// # Errors
    ///
    /// [`DiffError::Provider`] or [`DiffError::Flatten`] when `P` cannot open, read or flatten a
    /// fixture.
    pub fn run<P: DwgProvider>(&mut self, fixtures: &[Fixture]) -> Result<(), DiffError> {
        let mut name = String::new();
        let mut out = Vec::new();
        for fx in fixtures {
            let p = P::open(CadSource::Path(&fx.path)).map_err(|source| DiffError::Provider {
                provider: "(unopened)".into(),
                fixture: fx.id.clone(),
                source,
            })?;
            name = p.name().to_string();
            let fail = |source| DiffError::Provider {
                provider: p.name().to_string(),
                fixture: fx.id.clone(),
                source,
            };
            let layout = p
                .layouts()
                .map_err(fail)?
                .into_iter()
                .find(|l| l.kind == LayoutKind::Model)
                .ok_or_else(|| DiffError::Flatten {
                    provider: name.clone(),
                    fixture: fx.id.clone(),
                    detail: "the drawing has no model space".into(),
                })?;
            let flat = BlockResolver::new(&p, FlattenBounds::default())
                .flatten(
                    &EntityScope::Layout(layout.id),
                    &fx.id,
                    &layout.name,
                    &mut NoProvenance,
                    &mut LineageStore::new(),
                )
                .map_err(|e| DiffError::Flatten {
                    provider: name.clone(),
                    fixture: fx.id.clone(),
                    detail: e.to_string(),
                })?;
            let mut seen: BTreeMap<String, u32> = BTreeMap::new();
            let mut unique = |base: String| {
                let n = seen.entry(base.clone()).or_insert(0);
                *n += 1;
                if *n == 1 {
                    base
                } else {
                    format!("{base}~{}", *n - 1)
                }
            };
            let mut rows = Vec::new();
            for (i, f) in flat.entities.iter().enumerate() {
                let handle = f
                    .entity
                    .provenance
                    .handle
                    .map_or_else(|| format!("n{i}"), |h| format!("{:X}", h.get()));
                let key = unique(format!("{}#{handle}", f.entity.provenance.block_path));
                rows.push(row_of(f, key));
            }
            for s in &flat.skipped {
                let handle = s
                    .insert_handle
                    .map_or_else(|| "-".to_string(), |h| format!("{:X}", h.get()));
                rows.push(EntityRow {
                    key: unique(format!("{}#{handle}", s.block_path)),
                    class_name: "INSERT".into(),
                    layer: String::new(),
                    params: Vec::new(),
                    bucket: s.class.tag().to_string(),
                    reason: s.reason.code().to_string(),
                    pvec_type: String::new(),
                });
            }
            out.push(FixtureRun {
                fixture: fx.id.clone(),
                evidence: fx.evidence.clone(),
                rows,
            });
        }
        self.runs.push(ProviderRun {
            provider: name,
            fixtures: out,
        });
        Ok(())
    }

    /// Records a provider that is not compiled in or has no adapter, with why.
    pub fn absent(&mut self, provider: impl Into<String>, reason: impl Into<String>) {
        self.absent.push((provider.into(), reason.into()));
    }

    /// The rows `provider` produced for `fixture`.
    #[must_use]
    pub fn rows(&self, provider: &str, fixture: &str) -> Option<&[EntityRow]> {
        self.runs
            .iter()
            .find(|r| r.provider == provider)?
            .fixtures
            .iter()
            .find(|f| f.fixture == fixture)
            .map(|f| f.rows.as_slice())
    }

    /// Every difference between the first provider that ran and each of the others, ordered by
    /// provider, fixture, entity and field.
    #[must_use]
    pub fn differences(&self) -> Vec<Difference> {
        let Some((base, others)) = self.runs.split_first() else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for other in others {
            for bf in &base.fixtures {
                let Some(of) = other.fixtures.iter().find(|f| f.fixture == bf.fixture) else {
                    continue;
                };
                let by_key = |rows: &'_ [EntityRow]| -> BTreeMap<String, EntityRow> {
                    rows.iter().map(|r| (r.key.clone(), r.clone())).collect()
                };
                let (l, r) = (by_key(&bf.rows), by_key(&of.rows));
                let keys: BTreeSet<&String> = l.keys().chain(r.keys()).collect();
                for k in keys {
                    let mut push = |field: &str, left: String, right: String| {
                        if left != right {
                            out.push(Difference {
                                fixture: bf.fixture.clone(),
                                entity: k.clone(),
                                field: field.to_string(),
                                left_provider: base.provider.clone(),
                                left,
                                right_provider: other.provider.clone(),
                                right,
                            });
                        }
                    };
                    match (l.get(k), r.get(k)) {
                        (Some(a), Some(b)) => {
                            push("class_name", a.class_name.clone(), b.class_name.clone());
                            push("layer", a.layer.clone(), b.layer.clone());
                            push("bucket", a.bucket.clone(), b.bucket.clone());
                            push("reason", a.reason.clone(), b.reason.clone());
                            push("pvec_type", a.pvec_type.clone(), b.pvec_type.clone());
                            let pa: BTreeMap<_, _> = a.params.iter().cloned().collect();
                            let pb: BTreeMap<_, _> = b.params.iter().cloned().collect();
                            let names: BTreeSet<&String> = pa.keys().chain(pb.keys()).collect();
                            for n in names {
                                let show = |m: &BTreeMap<String, f64>| {
                                    m.get(n)
                                        .map_or_else(|| "absent".to_string(), |v| format!("{v:?}"))
                                };
                                let same = match (pa.get(n), pb.get(n)) {
                                    (Some(x), Some(y)) => x.to_bits() == y.to_bits(),
                                    (None, None) => true,
                                    _ => false,
                                };
                                if !same {
                                    push(&format!("param:{n}"), show(&pa), show(&pb));
                                }
                            }
                        }
                        (a, b) => push(
                            "present",
                            if a.is_some() { "yes" } else { "no" }.to_string(),
                            if b.is_some() { "yes" } else { "no" }.to_string(),
                        ),
                    }
                }
            }
        }
        out
    }

    /// The verdict against `deltas`.
    #[must_use]
    pub fn verdict(&self, deltas: &Deltas) -> Verdict {
        if self.runs.len() < 2 || !self.absent.is_empty() {
            let mut missing: Vec<String> = self.absent.iter().map(|(n, _)| n.clone()).collect();
            if missing.is_empty() {
                missing.push("a second provider".to_string());
            }
            return Verdict::Partial { missing };
        }
        let diffs = self.differences();
        let listed = |d: &Difference, a: &AllowedDelta| {
            a.fixture == d.fixture && a.entity == d.entity && a.field == d.field
        };
        let unlisted: Vec<Difference> = diffs
            .iter()
            .filter(|d| !deltas.entries.iter().any(|a| listed(d, a)))
            .cloned()
            .collect();
        let stale: Vec<AllowedDelta> = deltas
            .entries
            .iter()
            .filter(|a| !diffs.iter().any(|d| listed(d, a)))
            .cloned()
            .collect();
        if unlisted.is_empty() && stale.is_empty() {
            Verdict::Equivalent
        } else {
            Verdict::Failed { unlisted, stale }
        }
    }

    /// The report: one `result` line per provider and fixture, the absent providers, the
    /// differences, and the verdict. The same inputs give the same text.
    #[must_use]
    pub fn report(&self, deltas: &Deltas) -> String {
        let platform = platform();
        let mut o = String::from("provider differential report\n");
        for run in &self.runs {
            for f in &run.fixtures {
                let _ = writeln!(
                    o,
                    "result provider={} fixture={} evidence={} entities={} platform={platform} corpus={}",
                    run.provider,
                    f.fixture,
                    f.evidence,
                    f.rows.len(),
                    self.corpus
                );
            }
        }
        for (name, reason) in &self.absent {
            let _ = writeln!(o, "absent provider={name} reason={reason}");
        }
        for d in self.differences() {
            let _ = writeln!(
                o,
                "difference fixture={} entity={} field={} {}={} {}={}",
                d.fixture, d.entity, d.field, d.left_provider, d.left, d.right_provider, d.right
            );
        }
        let v = self.verdict(deltas);
        match &v {
            Verdict::Partial { missing } => {
                let _ = writeln!(
                    o,
                    "verdict PARTIAL missing={}: no differential comparison was performed",
                    missing.join(",")
                );
            }
            Verdict::Failed { unlisted, stale } => {
                let _ = writeln!(
                    o,
                    "verdict FAILED unlisted={} stale={}",
                    unlisted.len(),
                    stale.len()
                );
                for s in stale {
                    let _ = writeln!(
                        o,
                        "stale fixture={} entity={} field={} reason={}",
                        s.fixture, s.entity, s.field, s.reason
                    );
                }
            }
            Verdict::Equivalent => {
                for f in self
                    .runs
                    .iter()
                    .flat_map(|r| &r.fixtures)
                    .map(|f| (&f.fixture, &f.evidence))
                    .collect::<BTreeSet<_>>()
                {
                    let _ = writeln!(
                        o,
                        "agreement fixture={} evidence={} agreement={AGREEMENT_CLASS}",
                        f.0, f.1
                    );
                }
                let _ = writeln!(o, "verdict EQUIVALENT");
            }
        }
        o
    }
}
