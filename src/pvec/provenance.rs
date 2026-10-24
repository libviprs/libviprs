//! Source provenance records (issue #1096, N3.2; decision in `docs/adr/pvec-provenance.md`).
//!
//! Provenance answers "which DWG entity produced this curve" and it lives beside the geometry,
//! not inside it. [`Normalized2DGeometry`](super::Normalized2DGeometry) holds at most a
//! [`ProvenanceHandle`], which is a dense index into a [`ProvenanceStore`], so a conversion
//! that runs with provenance off allocates no [`Provenance`] at all and produces the same
//! geometry bytes as one that runs with it on.
//!
//! # The sink
//!
//! Producers (the normaliser, and N4.4's transform lineage when it composes through blocks)
//! talk to a [`ProvenanceSink`]. [`ProvenanceStore`] keeps everything and [`NoProvenance`]
//! keeps nothing. `record` takes a closure, so the off sink never runs the code that clones
//! names and paths into a record.
//!
//! # Bounds
//!
//! Every string a record holds is cut at [`MAX_NAME_BYTES`] (on a character boundary), a block
//! path at [`MAX_BLOCK_PATH_DEPTH`] steps and a warning list at [`MAX_WARNINGS`] entries of at
//! most [`MAX_WARNING_BYTES`] each. A cut sets [`Provenance::is_truncated`]. The bounds apply
//! when a record is built, so a hostile name in a drawing costs at most the bound, whatever its
//! length. There is no arithmetic on coordinates here: the transform is stored as extracted.

use std::fmt;

use super::normalized::ProvenanceHandle;
use super::provider::{InstancePath, ProvenanceStub, SourceTransform};
use crate::cad::ItemHandle;

/// The longest any one name (source, layout, class, layer, block) is kept, in bytes.
pub const MAX_NAME_BYTES: usize = 256;
/// The deepest block path kept. Deeper steps are dropped and the record is marked truncated.
pub const MAX_BLOCK_PATH_DEPTH: usize = 64;
/// How many warnings one record keeps.
pub const MAX_WARNINGS: usize = 16;
/// The longest any one warning is kept, in bytes.
pub const MAX_WARNING_BYTES: usize = 512;

/// `s` cut to at most `max` bytes on a character boundary, and whether it was cut.
fn bound(s: &str, max: usize) -> (String, bool) {
    if s.len() <= max {
        return (s.to_owned(), false);
    }
    let mut end = max;
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    (s[..end].to_owned(), true)
}

/// Where one normalised entity came from.
#[derive(Debug, Clone)]
pub struct Provenance {
    source: String,
    layout: String,
    handle: Option<ItemHandle>,
    class_name: String,
    block_path: InstancePath,
    layer: String,
    transform: SourceTransform,
    warnings: Vec<String>,
    truncated: bool,
}

impl Provenance {
    /// Builds a record from the provider's stub plus what the stub does not carry: the source
    /// file identity, the page or layout, and the transform as extracted. Everything is bounded.
    #[must_use]
    pub fn from_stub(
        stub: &ProvenanceStub,
        source: &str,
        layout: &str,
        transform: SourceTransform,
    ) -> Self {
        let mut truncated = false;
        let mut cut = |s: &str| {
            let (v, t) = bound(s, MAX_NAME_BYTES);
            truncated |= t;
            v
        };
        let source = cut(source);
        let layout = cut(layout);
        let class_name = cut(&stub.class_name);
        let layer = cut(&stub.layer);
        let mut block_path = InstancePath::root();
        for step in stub.block_path.steps().iter().take(MAX_BLOCK_PATH_DEPTH) {
            block_path = block_path.child(cut(&step.block_name), step.insert_handle);
        }
        truncated |= stub.block_path.depth() > MAX_BLOCK_PATH_DEPTH;
        Provenance {
            source,
            layout,
            handle: stub.handle,
            class_name,
            block_path,
            layer,
            transform,
            warnings: Vec::new(),
            truncated,
        }
    }

    /// The source file identity.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }

    /// The page or layout the entity was read from.
    #[must_use]
    pub fn layout(&self) -> &str {
        &self.layout
    }

    /// The entity's DWG handle, when the drawing gives one.
    #[must_use]
    pub fn handle(&self) -> Option<ItemHandle> {
        self.handle
    }

    /// The drawing's class name for the entity.
    #[must_use]
    pub fn class_name(&self) -> &str {
        &self.class_name
    }

    /// The INSERT chain from the layout down to the entity, outermost first.
    #[must_use]
    pub fn block_path(&self) -> &InstancePath {
        &self.block_path
    }

    /// The entity's own layer name.
    #[must_use]
    pub fn layer(&self) -> &str {
        &self.layer
    }

    /// The transform as extracted, uncomposed.
    #[must_use]
    pub fn transform(&self) -> &SourceTransform {
        &self.transform
    }

    /// Warnings raised while mapping this entity, oldest first.
    #[must_use]
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Whether any field was cut, or any warning cut or dropped, to stay inside the bounds.
    #[must_use]
    pub fn is_truncated(&self) -> bool {
        self.truncated
    }

    fn add_warning(&mut self, message: &str) {
        if self.warnings.len() >= MAX_WARNINGS {
            self.truncated = true;
            return;
        }
        let (w, cut) = bound(message, MAX_WARNING_BYTES);
        self.truncated |= cut;
        self.warnings.push(w);
    }

    /// Writes the record as one line, for the conversion report and the N3.8 dump. Strings are
    /// escaped, so a record never spans lines whatever a drawing put in a name.
    pub fn write_report<W: fmt::Write>(&self, w: &mut W) -> fmt::Result {
        write!(w, "source={:?} layout={:?}", self.source, self.layout)?;
        match self.handle {
            Some(h) => write!(w, " handle={:X}", h.get())?,
            None => w.write_str(" handle=-")?,
        }
        write!(
            w,
            " class={:?} layer={:?} path=",
            self.class_name, self.layer
        )?;
        if self.block_path.is_root() {
            w.write_char('/')?;
        }
        for (i, step) in self.block_path.steps().iter().enumerate() {
            if i > 0 {
                w.write_char('/')?;
            }
            write!(
                w,
                "{}@{:X}",
                step.block_name.escape_debug(),
                step.insert_handle.get()
            )?;
        }
        write!(w, " xf={:?}", self.transform.coefficients())?;
        write!(w, " warnings={:?}", self.warnings)?;
        if self.truncated {
            w.write_str(" truncated")?;
        }
        Ok(())
    }
}

/// Where a producer sends provenance.
pub trait ProvenanceSink {
    /// Records one entity's provenance and returns the handle to attach to its geometry record,
    /// or `None` when this sink keeps nothing. `build` runs only when something is kept.
    fn record<F: FnOnce() -> Provenance>(&mut self, build: F) -> Option<ProvenanceHandle>;

    /// Attaches a warning to the record `handle` names. A `None` handle is a no-op, so a
    /// producer can pass through whatever `record` gave it.
    fn warn(&mut self, handle: Option<ProvenanceHandle>, message: &str);
}

/// The off switch: keeps nothing and never runs the builder.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoProvenance;

impl ProvenanceSink for NoProvenance {
    fn record<F: FnOnce() -> Provenance>(&mut self, _build: F) -> Option<ProvenanceHandle> {
        None
    }

    fn warn(&mut self, _handle: Option<ProvenanceHandle>, _message: &str) {}
}

/// The side table: records keyed by a dense index, which is the handle's value.
#[derive(Debug, Clone, Default)]
pub struct ProvenanceStore {
    records: Vec<Provenance>,
}

impl ProvenanceStore {
    /// An empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// How many records it holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Whether it holds none.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// The record a handle names, `None` for a handle this store did not issue.
    #[must_use]
    pub fn get(&self, handle: ProvenanceHandle) -> Option<&Provenance> {
        self.records.get(usize::try_from(handle.get()).ok()?)
    }

    /// Writes every record as one line each, in index order.
    pub fn write_report<W: fmt::Write>(&self, w: &mut W) -> fmt::Result {
        for r in &self.records {
            r.write_report(w)?;
            w.write_char('\n')?;
        }
        Ok(())
    }
}

impl ProvenanceSink for ProvenanceStore {
    fn record<F: FnOnce() -> Provenance>(&mut self, build: F) -> Option<ProvenanceHandle> {
        let index = self.records.len() as u64;
        self.records.push(build());
        Some(ProvenanceHandle::new(index))
    }

    fn warn(&mut self, handle: Option<ProvenanceHandle>, message: &str) {
        let Some(h) = handle else { return };
        if let Some(r) = usize::try_from(h.get())
            .ok()
            .and_then(|i| self.records.get_mut(i))
        {
            r.add_warning(message);
        }
    }
}
