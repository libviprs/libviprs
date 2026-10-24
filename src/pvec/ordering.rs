//! Deterministic source ordering and file-local identity (issue #1102, N3.7).
//!
//! One drawing has to convert to the same output every time, so neither entity order nor
//! entity identity may depend on a hash map's iteration order or on how a decoder happened to
//! walk the file. This module is the single place that decides both.
//!
//! # Two bases
//!
//! * [`OrderingBasis::DecoderTraversal`]: the provider documents its traversal as
//!   deterministic, and the arrival order is used as is.
//! * [`OrderingBasis::Canonicalised`]: libviprs sorts the entities itself by [`canonical_cmp`]
//!   before anything is emitted, so the result does not depend on traversal order at all.
//!
//! Which one applies is looked up per provider and provider version in an [`OrderingPolicy`].
//! Nothing is declared by default, so an unknown provider or a version nobody has vetted is
//! canonicalised. A provider upgrade therefore falls back to the safe basis, and the choice is
//! written into the conversion report ([`OrderingReport`]) so it is visible rather than silent.
//!
//! # The canonical total order
//!
//! Entities compare by, in turn:
//!
//! 1. block path, step by step on (block name, INSERT handle), a shorter path before any
//!    longer path it is a prefix of (so the layout's own entities come first);
//! 2. entity handle, `None` before any `Some`;
//! 3. the entity's index (its position in the traversal it arrived in);
//! 4. tie-break, only when 1 to 3 are all equal: class name, then layer name, then the
//!    `Debug` rendering of the geometry. Two entities equal on all of that are
//!    indistinguishable, so their relative order cannot show in the output.
//!
//! # Identity
//!
//! [`EntityId`] is the entity's position in the final order, dense from zero. It is a pure
//! function of the entities, so it does not read a provenance sink and a provenance-off
//! conversion yields the same ids. Ids are file-local: two files share no identity space.
//!
//! No `HashMap` or `HashSet` is used anywhere in `src/pvec`; `tests/pvec_1102_ordering.rs`
//! fails if one is introduced.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;

use super::provider::RawEntity;

/// Where an ordering came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OrderingBasis {
    /// The provider's documented traversal order, used as it arrived.
    DecoderTraversal,
    /// libviprs' own total order, [`canonical_cmp`].
    Canonicalised,
}

impl OrderingBasis {
    /// The stable report name.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            OrderingBasis::DecoderTraversal => "decoder-traversal",
            OrderingBasis::Canonicalised => "canonicalised",
        }
    }
}

impl fmt::Display for OrderingBasis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Which provider versions are declared to traverse deterministically.
#[derive(Debug, Clone, Default)]
pub struct OrderingPolicy {
    declared: BTreeMap<(String, String), OrderingBasis>,
}

impl OrderingPolicy {
    /// A policy that declares nothing, so everything is canonicalised.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Declares the basis for one provider version.
    #[must_use]
    pub fn declare(mut self, provider: &str, version: &str, basis: OrderingBasis) -> Self {
        self.declared
            .insert((provider.to_owned(), version.to_owned()), basis);
        self
    }

    /// The basis for a provider version, [`OrderingBasis::Canonicalised`] when undeclared.
    #[must_use]
    pub fn basis_for(&self, provider: &str, version: &str) -> OrderingBasis {
        self.declared
            .get(&(provider.to_owned(), version.to_owned()))
            .copied()
            .unwrap_or(OrderingBasis::Canonicalised)
    }
}

/// A file-local entity id: the entity's position in the final order, dense from zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EntityId(u64);

impl EntityId {
    /// The raw id.
    #[must_use]
    pub fn get(self) -> u64 {
        self.0
    }
}

/// An entity with its id.
#[derive(Debug, Clone)]
pub struct OrderedEntity {
    /// The file-local id.
    pub id: EntityId,
    /// The entity.
    pub entity: RawEntity,
}

/// What ordering was applied to one source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderingReport {
    provider: String,
    version: String,
    basis: OrderingBasis,
    entities: u64,
}

impl OrderingReport {
    /// The provider name.
    #[must_use]
    pub fn provider(&self) -> &str {
        &self.provider
    }

    /// The provider version.
    #[must_use]
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Whether the order came from the decoder or from libviprs.
    #[must_use]
    pub fn basis(&self) -> OrderingBasis {
        self.basis
    }

    /// How many entities were ordered.
    #[must_use]
    pub fn entities(&self) -> u64 {
        self.entities
    }

    /// Writes the report as one line, strings escaped.
    ///
    /// # Errors
    ///
    /// Whatever the writer returns.
    pub fn write_report<W: fmt::Write>(&self, w: &mut W) -> fmt::Result {
        write!(
            w,
            "ordering provider={:?} version={:?} basis={} entities={}",
            self.provider, self.version, self.basis, self.entities
        )
    }
}

/// Entities in final order, numbered, with the report of how.
#[derive(Debug, Clone)]
pub struct OrderedSource {
    /// The entities, in emit order.
    pub entities: Vec<OrderedEntity>,
    /// How the order was decided.
    pub report: OrderingReport,
}

/// The canonical total order over `(index, entity)` pairs; see the module docs for the keys.
#[must_use]
pub fn canonical_cmp(ia: u64, a: &RawEntity, ib: u64, b: &RawEntity) -> Ordering {
    let (pa, pb) = (&a.provenance, &b.provenance);
    let path = |p: &super::provider::InstancePath| -> Vec<(String, u64)> {
        p.steps()
            .iter()
            .map(|s| (s.block_name.clone(), s.insert_handle.get()))
            .collect()
    };
    path(&pa.block_path)
        .cmp(&path(&pb.block_path))
        .then_with(|| pa.handle.cmp(&pb.handle))
        .then_with(|| ia.cmp(&ib))
        .then_with(|| pa.class_name.cmp(&pb.class_name))
        .then_with(|| pa.layer.cmp(&pb.layer))
        .then_with(|| format!("{:?}", a.geometry).cmp(&format!("{:?}", b.geometry)))
}

/// Orders and numbers one source's entities.
///
/// `entities` is the provider's traversal in arrival order; the index in the canonical order
/// is the arrival position. Under [`OrderingBasis::DecoderTraversal`] the arrival order is
/// kept, under [`OrderingBasis::Canonicalised`] it is sorted by [`canonical_cmp`].
#[must_use]
pub fn order_entities(
    provider: &str,
    version: &str,
    policy: &OrderingPolicy,
    entities: Vec<RawEntity>,
) -> OrderedSource {
    let basis = policy.basis_for(provider, version);
    let count = entities.len() as u64;
    let mut indexed: Vec<(u64, RawEntity)> = entities
        .into_iter()
        .enumerate()
        .map(|(i, e)| (i as u64, e))
        .collect();
    if basis == OrderingBasis::Canonicalised {
        indexed.sort_by(|(ia, a), (ib, b)| canonical_cmp(*ia, a, *ib, b));
    }
    let entities = indexed
        .into_iter()
        .enumerate()
        .map(|(n, (_, entity))| OrderedEntity {
            id: EntityId(n as u64),
            entity,
        })
        .collect();
    OrderedSource {
        entities,
        report: OrderingReport {
            provider: provider.to_owned(),
            version: version.to_owned(),
            basis,
            entities: count,
        },
    }
}
