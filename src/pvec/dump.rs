//! The debug and inspection export (issue #1103, N3.8). Stub: red tests first.

use std::fmt;

use super::normalized::Normalized2DGeometry;
use super::provenance::{Provenance, ProvenanceStore};

/// The format version on the dump's first line.
pub const DUMP_VERSION: u32 = 1;

/// A streaming writer over normalized entities.
#[derive(Debug)]
pub struct NormalizedDump<W: fmt::Write> {
    out: W,
    count: u64,
}

impl<W: fmt::Write> NormalizedDump<W> {
    /// Starts a dump.
    pub fn new(out: W) -> Result<Self, fmt::Error> {
        Ok(NormalizedDump { out, count: 0 })
    }

    /// Writes one entity.
    pub fn push(
        &mut self,
        _entity: &Normalized2DGeometry,
        _provenance: Option<&Provenance>,
    ) -> Result<u64, fmt::Error> {
        let id = self.count;
        self.count += 1;
        Ok(id)
    }

    /// Writes one entity, resolving its provenance handle against `store`.
    pub fn push_stored(
        &mut self,
        entity: &Normalized2DGeometry,
        store: &ProvenanceStore,
    ) -> Result<u64, fmt::Error> {
        self.push(entity, entity.provenance().and_then(|h| store.get(h)))
    }

    /// How many entities have been written.
    pub fn entities_written(&self) -> u64 {
        self.count
    }

    /// Closes the dump and hands the sink back.
    pub fn finish(self) -> Result<W, fmt::Error> {
        Ok(self.out)
    }
}

/// Dumps `entities` to a string.
pub fn dump_to_string<'a, I>(_entities: I) -> String
where
    I: IntoIterator<Item = (&'a Normalized2DGeometry, Option<&'a Provenance>)>,
{
    String::new()
}
