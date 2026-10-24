//! Deterministic source ordering and file-local identity (issue #1102, N3.7). Stub, see tests.

use std::cmp::Ordering;
use std::fmt;

use super::provider::RawEntity;

/// Where an ordering came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OrderingBasis {
    /// The provider's documented traversal order.
    DecoderTraversal,
    /// libviprs' own total order.
    Canonicalised,
}

/// Which provider versions are declared deterministic.
#[derive(Debug, Clone, Default)]
pub struct OrderingPolicy {}

impl OrderingPolicy {
    /// Empty policy.
    #[must_use]
    pub fn new() -> Self {
        todo!()
    }
    /// Declares a basis.
    #[must_use]
    pub fn declare(self, _provider: &str, _version: &str, _basis: OrderingBasis) -> Self {
        todo!()
    }
    /// The basis for a provider version.
    #[must_use]
    pub fn basis_for(&self, _provider: &str, _version: &str) -> OrderingBasis {
        todo!()
    }
}

/// A file-local entity id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EntityId(u64);

impl EntityId {
    /// Raw value.
    #[must_use]
    pub fn get(self) -> u64 {
        todo!()
    }
}

/// An entity with its id.
#[derive(Debug, Clone)]
pub struct OrderedEntity {
    /// Id.
    pub id: EntityId,
    /// Entity.
    pub entity: RawEntity,
}

/// What ordering was applied.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderingReport {}

impl OrderingReport {
    /// Basis.
    #[must_use]
    pub fn basis(&self) -> OrderingBasis {
        todo!()
    }
    /// One line.
    pub fn write_report<W: fmt::Write>(&self, _w: &mut W) -> fmt::Result {
        todo!()
    }
}

/// Ordered entities plus report.
#[derive(Debug, Clone)]
pub struct OrderedSource {
    /// Entities.
    pub entities: Vec<OrderedEntity>,
    /// Report.
    pub report: OrderingReport,
}

/// The canonical total order.
#[must_use]
pub fn canonical_cmp(_ia: u64, _a: &RawEntity, _ib: u64, _b: &RawEntity) -> Ordering {
    todo!()
}

/// Orders and numbers entities.
#[must_use]
pub fn order_entities(
    _provider: &str,
    _version: &str,
    _policy: &OrderingPolicy,
    _entities: Vec<RawEntity>,
) -> OrderedSource {
    todo!()
}
