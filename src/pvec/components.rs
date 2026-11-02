//! PVec's component streams: layer, line style and provenance (issue #1093, N6.6).
//!
//! Visual semantics and the source-entity audit trail stay out of the geometry. Each stream is
//! optional and independently omittable, and none is read by anything that only wants curves, so
//! a run with every stream off writes the same geometry blocks as a run with every stream on.
//!
//! # Ids
//!
//! A stream refers to an entity by PVec's file-local id, which is the position of the entity in the
//! artifact. [`build_streams`] takes the artifact's `source_ids` (element `i` is the libviprs id of
//! PVec entity `i`) and maps through it, so a row for an entity that carried no geometry and was
//! not written is dropped rather than pointing at a stranger.
//!
//! # Provenance
//!
//! One row per *source entity*, not per emitted curve. An entity reached through INSERTs belongs to
//! the outermost INSERT (the first step of its block path, which is the N4.4 lineage), so every
//! curve one INSERT instance flattens into shares one row, written against the first of them. An
//! entity on the layout itself is its own source entity. The row holds the source handle, since the
//! stream carries nothing else; the block path, layer and warnings are in the conversion report.

use std::collections::BTreeMap;

use pvec_capabilities::Registry;
use pvec_facade::EntityId as PvecEntityId;
use pvec_facade::components::Stream;
use thiserror::Error;

use super::ordering::EntityId;
use super::provider::{LayerState, ProvenanceStub};
use super::style::LineStyleComponent;

/// Which streams a run writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamSelection {
    /// The layer stream.
    pub layer: bool,
    /// The line style stream.
    pub line_style: bool,
    /// The provenance stream.
    pub provenance: bool,
}

impl StreamSelection {
    /// Every stream.
    pub const ALL: Self = Self {
        layer: true,
        line_style: true,
        provenance: true,
    };
    /// No stream: a geometry-only artifact.
    pub const NONE: Self = Self {
        layer: false,
        line_style: false,
        provenance: false,
    };
}

/// What the streams need to know about one source entity.
#[derive(Debug, Clone, Copy)]
pub struct ComponentSource<'a> {
    /// The entity's libviprs id.
    pub id: EntityId,
    /// Where it came from, which also names its layer.
    pub stub: &'a ProvenanceStub,
    /// Its line style, when the run computed one.
    pub style: Option<&'a LineStyleComponent>,
}

/// Why the streams could not be built.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ComponentsError {
    /// PVec refused a payload; the rule it names is the message.
    #[error("PVec refused a component stream: {0}")]
    Refused(&'static str),
}

/// The streams `select` asks for, built from `rows` and mapped onto the artifact written with
/// `source_ids`. A stream with no row is not emitted.
///
/// # Errors
///
/// [`ComponentsError::Refused`] when PVec's encoder refuses a payload (a name past its bound).
pub fn build_streams(
    select: StreamSelection,
    source_ids: &[EntityId],
    layers: &[LayerState],
    rows: &[ComponentSource<'_>],
) -> Result<Vec<Stream>, ComponentsError> {
    let _ = (select, source_ids, layers, rows, Registry::v1());
    let _: Option<PvecEntityId> = None;
    let _ = BTreeMap::<u64, u32>::new();
    todo!("N6.6")
}
