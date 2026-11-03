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

use std::collections::{BTreeMap, BTreeSet};

use pvec_capabilities::Registry;
use pvec_facade::EntityId as PvecEntityId;
use pvec_facade::components::{
    LAYER_FROZEN, LAYER_LOCKED, LAYER_OFF, LayerStream, LineStyleDef, LineStyleStream,
    ProvenanceRecord, ProvenanceStream, Stream, StrokeWidth,
};
use thiserror::Error;

use super::ordering::EntityId;
use super::provider::{LayerState, ProvenanceStub};
use super::style::{LineStyleComponent, StrokeWidth as StyleWidth};
use crate::cad::ItemHandle;

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

/// PVec's bounds on a component payload, repeated here because the facade does not export them.
/// `tests/pvec_1093_degrade.rs` holds each one against the encoder, one past it and at it.
const MAX_NAME: usize = 255;
const MAX_DASHES: usize = 16;
const MAX_TABLE: usize = 65_536;
const MAX_RECORDS: usize = 1 << 20;

/// A row or table entry left out of a stream because PVec cannot hold it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DroppedRow {
    /// The entity the row was for.
    pub entity: EntityId,
    /// The stream it would have gone in: `layer`, `line-style` or `provenance`.
    pub stream: &'static str,
    /// Why it was left out: `name-too-long`, `too-many-dashes`, `table-full` or `stream-full`.
    pub reason: &'static str,
}

/// The streams a run writes and the rows that did not fit in them.
#[derive(Debug)]
pub struct BuiltStreams {
    /// The streams, in the order layer, line style, provenance.
    pub streams: Vec<Stream>,
    /// Every row left out, for the conversion report.
    pub dropped: Vec<DroppedRow>,
}

/// Why the streams could not be built.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ComponentsError {
    /// PVec refused a payload; the rule it names is the message.
    #[error("PVec refused a component stream: {0}")]
    Refused(String),
}

/// The streams `select` asks for, built from `rows` and mapped onto the artifact written with
/// `source_ids`. A stream with no row is not emitted.
///
/// A row PVec cannot hold (a layer name past 255 bytes, a dash array past 16 entries, a table past
/// 65536 entries, a stream past 2^20 rows) is left out and listed in [`BuiltStreams::dropped`], so
/// one bad entity costs its own row and not every stream.
///
/// # Errors
///
/// [`ComponentsError::Refused`] when PVec's encoder refuses a payload for a reason not covered
/// above.
pub fn build_streams(
    select: StreamSelection,
    source_ids: &[EntityId],
    layers: &[LayerState],
    rows: &[ComponentSource<'_>],
) -> Result<BuiltStreams, ComponentsError> {
    let mut dropped = Vec::new();
    if select == StreamSelection::NONE {
        return Ok(BuiltStreams {
            streams: Vec::new(),
            dropped,
        });
    }
    let registry = Registry::v1();
    // The artifact's id for each libviprs id that was written.
    let pvec_ids: BTreeMap<EntityId, PvecEntityId> = source_ids
        .iter()
        .enumerate()
        .filter_map(|(n, id)| Some((*id, PvecEntityId::new(u32::try_from(n).ok()?).ok()?)))
        .collect();
    let written: Vec<(PvecEntityId, &ComponentSource<'_>)> = rows
        .iter()
        .filter_map(|r| Some((*pvec_ids.get(&r.id)?, r)))
        .collect();
    let mut out = Vec::new();
    let mut skip = |r: &ComponentSource<'_>, stream, reason| {
        dropped.push(DroppedRow {
            entity: r.id,
            stream,
            reason,
        });
    };
    if select.layer && !written.is_empty() {
        let mut names: Vec<String> = Vec::new();
        let mut index: BTreeMap<&str, u32> = BTreeMap::new();
        let mut assignments = Vec::with_capacity(written.len());
        for (id, r) in &written {
            let name = r.stub.layer.as_str();
            if name.len() > MAX_NAME {
                skip(r, "layer", "name-too-long");
                continue;
            }
            if assignments.len() >= MAX_RECORDS {
                skip(r, "layer", "stream-full");
                continue;
            }
            let n = match index.get(name) {
                Some(n) => *n,
                None => {
                    if names.len() >= MAX_TABLE {
                        skip(r, "layer", "table-full");
                        continue;
                    }
                    let n = u32::try_from(names.len()).unwrap_or(u32::MAX);
                    names.push(name.to_owned());
                    index.insert(name, n);
                    n
                }
            };
            assignments.push((*id, n));
        }
        let flags = names
            .iter()
            .map(|n| {
                layers.iter().find(|l| &l.name == n).map_or(0, |l| {
                    (if l.off { LAYER_OFF } else { 0 })
                        | (if l.frozen { LAYER_FROZEN } else { 0 })
                        | (if l.locked { LAYER_LOCKED } else { 0 })
                })
            })
            .collect();
        let stream = LayerStream {
            names,
            flags,
            assignments,
        };
        if !stream.assignments.is_empty() {
            out.push(
                stream
                    .into_stream(&registry)
                    .map_err(|e| ComponentsError::Refused(e.to_string()))?,
            );
        }
    }
    if select.line_style {
        let mut styles: Vec<LineStyleDef> = Vec::new();
        let mut assignments = Vec::new();
        for (id, r) in &written {
            let Some(c) = r.style else { continue };
            let def = LineStyleDef {
                name: String::new(),
                dashes: c
                    .dash
                    .as_ref()
                    .map_or_else(Vec::new, |d| d.array().to_vec()),
                width: c.width.map(|w| match w {
                    StyleWidth::Thinnest => StrokeWidth::Thinnest,
                    StyleWidth::Width(v) => StrokeWidth::Width(v),
                }),
                phase: c.dash.as_ref().map_or(0.0, |d| d.phase()),
                visible: c.visible,
            };
            if def.dashes.len() > MAX_DASHES {
                skip(r, "line-style", "too-many-dashes");
                continue;
            }
            if assignments.len() >= MAX_RECORDS {
                skip(r, "line-style", "stream-full");
                continue;
            }
            let n = match styles.iter().position(|s| *s == def) {
                Some(n) => n,
                None => {
                    if styles.len() >= MAX_TABLE {
                        skip(r, "line-style", "table-full");
                        continue;
                    }
                    styles.push(def);
                    styles.len() - 1
                }
            };
            assignments.push((*id, u32::try_from(n).unwrap_or(u32::MAX)));
        }
        if !assignments.is_empty() {
            let stream = LineStyleStream {
                styles,
                assignments,
            };
            out.push(
                stream
                    .into_stream(&registry)
                    .map_err(|e| ComponentsError::Refused(e.to_string()))?,
            );
        }
    }
    if select.provenance {
        // One row per source entity: the outermost INSERT when there is one, else the entity.
        let mut seen: BTreeSet<(Option<u64>, u64)> = BTreeSet::new();
        let mut records = Vec::new();
        for (id, r) in &written {
            let handle = match r.stub.block_path.steps().first() {
                Some(step) => Some(step.insert_handle.get()),
                None => r.stub.handle.map(ItemHandle::get),
            };
            // An entity with no handle is its own source entity, keyed by its id.
            let key = match handle {
                Some(h) => (Some(h), 0),
                None => (None, r.id.get()),
            };
            if !seen.insert(key) {
                continue;
            }
            if records.len() >= MAX_RECORDS {
                skip(r, "provenance", "stream-full");
                continue;
            }
            records.push(ProvenanceRecord {
                entity: *id,
                // 1 is a DWG source entity.
                source_kind: 1,
                handle: handle.map_or_else(Vec::new, |h| h.to_be_bytes().to_vec()),
            });
        }
        if !records.is_empty() {
            let stream = ProvenanceStream { records };
            out.push(
                stream
                    .into_stream(&registry)
                    .map_err(|e| ComponentsError::Refused(e.to_string()))?,
            );
        }
    }
    Ok(BuiltStreams {
        streams: out,
        dropped,
    })
}
