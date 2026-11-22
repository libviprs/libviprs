//! Xref resolution (#1631): the drawings a caller supplies for a host's xref blocks, served as if
//! they were block definitions of the host, so the view pass and the flatten pass walk them with
//! the rules they already apply (layer rules, the shared limits, depth, composed transforms).
//!
//! An xref's model space is the block `NAME`; a block of that drawing is `NAME|BLOCK`; its layers
//! are `NAME|LAYER` (layer `0` and `Defpoints` keep their name, as AutoCAD keeps them). A host
//! layer table row of the same name wins over the xref's own, because that row is the one a user
//! sees and switches. Nested xrefs prefix again: `NAME|INNER|LAYER`.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

use crate::cad::{CadSource, ItemHandle};
use crate::pvec::{
    Annotation, DwgProvider, EntityCursor, EntityFacts, EntityScope, InstancePath, LayerState,
    LayoutId, LayoutKind, LayoutPlot, LayoutState, ProviderCaps, ProviderError, RawEntity,
    RawGeometry, SourceFact, TextStyleRef, ViewportEntity, XrefAttachment,
};

/// What a conversion learned about its xrefs, for the report.
#[derive(Debug, Default)]
pub(super) struct XrefNote {
    /// The xref blocks whose drawing was entered, spelled as the walk reaches them.
    pub resolved: BTreeSet<String>,
    /// Supplied names that no xref block record of any drawing carries.
    pub unattached: Vec<String>,
}

struct Node<P> {
    key: String,
    provider: P,
}

/// The host drawing with the supplied xref drawings behind it.
pub(super) struct XrefProvider<P> {
    host: P,
    nodes: Vec<Node<P>>,
    note: Arc<Mutex<XrefNote>>,
}

/// Opens the supplied drawings the host's xref block records name, and theirs in turn, down to one
/// level past `max_depth` so the walk's own depth limit is what refuses a nesting that is too deep.
pub(super) fn attach<P: DwgProvider>(
    host: P,
    supplied: &[(String, CadSource<'_>)],
    max_depth: u32,
    note: Arc<Mutex<XrefNote>>,
) -> Result<XrefProvider<P>, ProviderError> {
    fn scan<P: DwgProvider>(
        provider: &P,
        prefix: Option<&str>,
        depth: u32,
        max_depth: u32,
        supplied: &[(String, CadSource<'_>)],
        matched: &mut [bool],
        nodes: &mut Vec<Node<P>>,
    ) -> Result<(), ProviderError> {
        if depth > max_depth {
            return Ok(());
        }
        for record in provider.xref_blocks().unwrap_or_default() {
            let Some(i) = supplied
                .iter()
                .position(|(n, _)| n.eq_ignore_ascii_case(&record.block_name))
            else {
                continue;
            };
            matched[i] = true;
            let child = P::open(supplied[i].1)?;
            let key = match prefix {
                Some(p) => format!("{p}|{}", record.block_name),
                None => record.block_name.clone(),
            };
            scan(
                &child,
                Some(&key),
                depth + 1,
                max_depth,
                supplied,
                matched,
                nodes,
            )?;
            nodes.push(Node {
                key,
                provider: child,
            });
        }
        Ok(())
    }
    let mut matched = vec![false; supplied.len()];
    let mut nodes = Vec::new();
    scan(
        &host,
        None,
        1,
        max_depth.saturating_add(1),
        supplied,
        &mut matched,
        &mut nodes,
    )?;
    // A name is attached when any opened drawing, the host included, has an xref block of that name.
    let mut records: Vec<XrefAttachment> = host.xref_blocks().unwrap_or_default();
    for n in &nodes {
        records.extend(n.provider.xref_blocks().unwrap_or_default());
    }
    let unattached = supplied
        .iter()
        .filter(|(name, _)| {
            !records
                .iter()
                .any(|r| r.block_name.eq_ignore_ascii_case(name))
        })
        .map(|(name, _)| name.clone())
        .collect();
    note.lock().expect("the note is not poisoned").unattached = unattached;
    Ok(XrefProvider { host, nodes, note })
}

fn keeps_name(layer: &str) -> bool {
    layer == "0" || layer.eq_ignore_ascii_case("Defpoints")
}

/// Where a scope is served from: the provider, the scope there, and the xref's key when it is one.
struct Routed<'s, P> {
    provider: &'s P,
    scope: EntityScope,
    key: Option<&'s str>,
    /// The INSERT chain to stamp on an xref's model space, which the provider read as a layout and
    /// so did not stamp.
    stamp: Option<InstancePath>,
}

impl<P: DwgProvider> XrefProvider<P> {
    fn route(&self, scope: &EntityScope) -> Result<Routed<'_, P>, ProviderError> {
        let EntityScope::BlockDefinition { name, reached_via } = scope else {
            return Ok(Routed {
                provider: &self.host,
                scope: scope.clone(),
                key: None,
                stamp: None,
            });
        };
        if let Some(n) = self.nodes.iter().find(|n| n.key.eq_ignore_ascii_case(name)) {
            let model = n
                .provider
                .layouts()?
                .into_iter()
                .find(|l| l.kind == LayoutKind::Model)
                .ok_or_else(|| ProviderError::NoSuchScope(format!("model space of {name}")))?;
            self.note
                .lock()
                .expect("the note is not poisoned")
                .resolved
                .insert(n.key.clone());
            return Ok(Routed {
                provider: &n.provider,
                scope: EntityScope::Layout(model.id),
                key: Some(&n.key),
                stamp: Some(reached_via.clone()),
            });
        }
        let inside = self
            .nodes
            .iter()
            .filter(|n| {
                name.len() > n.key.len()
                    && name.is_char_boundary(n.key.len())
                    && name[..n.key.len()].eq_ignore_ascii_case(&n.key)
                    && name[n.key.len()..].starts_with('|')
            })
            .max_by_key(|n| n.key.len());
        Ok(match inside {
            Some(n) => Routed {
                provider: &n.provider,
                scope: EntityScope::BlockDefinition {
                    name: name[n.key.len() + 1..].to_owned(),
                    reached_via: reached_via.clone(),
                },
                key: Some(&n.key),
                stamp: None,
            },
            None => Routed {
                provider: &self.host,
                scope: scope.clone(),
                key: None,
                stamp: None,
            },
        })
    }
}

/// An xref's entities with the names of the host's namespace.
struct Prefixed<'a> {
    inner: Box<dyn EntityCursor + 'a>,
    key: Option<&'a str>,
    stamp: Option<InstancePath>,
}

impl EntityCursor for Prefixed<'_> {
    fn next_entity(&mut self) -> Result<Option<RawEntity>, ProviderError> {
        let Some(mut e) = self.inner.next_entity()? else {
            return Ok(None);
        };
        if let Some(path) = &self.stamp {
            e.provenance.block_path = path.clone();
        }
        if let Some(key) = self.key {
            if !keeps_name(&e.provenance.layer) {
                e.provenance.layer = format!("{key}|{}", e.provenance.layer);
            }
            if let RawGeometry::Insert(r) = &mut e.geometry {
                r.block_name = format!("{key}|{}", r.block_name);
            }
        }
        Ok(Some(e))
    }
}

impl<P: DwgProvider> DwgProvider for XrefProvider<P> {
    fn probe(_: CadSource<'_>) -> Result<bool, ProviderError> {
        Ok(false)
    }

    fn open(_: CadSource<'_>) -> Result<Self, ProviderError> {
        Err(ProviderError::Open {
            provider: "xref",
            detail: "a host with its xrefs is built from the host, not opened".into(),
        })
    }

    fn name(&self) -> &'static str {
        self.host.name()
    }

    fn caps(&self) -> &ProviderCaps {
        self.host.caps()
    }

    fn layouts(&self) -> Result<Vec<LayoutState>, ProviderError> {
        self.host.layouts()
    }

    fn layout_plot(&self, layout: LayoutId) -> Result<SourceFact<LayoutPlot>, ProviderError> {
        self.host.layout_plot(layout)
    }

    fn layers(&self) -> Result<Vec<LayerState>, ProviderError> {
        let mut out = self.host.layers()?;
        let mut seen: BTreeSet<String> = out.iter().map(|l| l.name.to_ascii_lowercase()).collect();
        for n in &self.nodes {
            for l in n.provider.layers()? {
                if keeps_name(&l.name) {
                    continue;
                }
                let name = format!("{}|{}", n.key, l.name);
                if seen.insert(name.to_ascii_lowercase()) {
                    out.push(LayerState {
                        name,
                        handle: None,
                        ..l
                    });
                }
            }
        }
        Ok(out)
    }

    fn entity_facts(
        &self,
        scope: &EntityScope,
    ) -> Result<BTreeMap<ItemHandle, EntityFacts>, ProviderError> {
        let r = self.route(scope)?;
        r.provider.entity_facts(&r.scope)
    }

    fn viewports(&self) -> Result<Vec<ViewportEntity>, ProviderError> {
        self.host.viewports()
    }

    fn xref_blocks(&self) -> Result<Vec<XrefAttachment>, ProviderError> {
        Ok(self
            .host
            .xref_blocks()?
            .into_iter()
            .filter(|x| {
                !self
                    .nodes
                    .iter()
                    .any(|n| n.key.eq_ignore_ascii_case(&x.block_name))
            })
            .collect())
    }

    fn text_styles(
        &self,
        scope: &EntityScope,
    ) -> Result<BTreeMap<ItemHandle, TextStyleRef>, ProviderError> {
        let r = self.route(scope)?;
        r.provider.text_styles(&r.scope)
    }

    // Handles are per drawing and these three are asked by handle alone, so an xref's text is
    // answered from the host's table: an xref's own text keeps no anchor or annotation.
    fn text_anchor(&self, handle: ItemHandle) -> Option<(String, [f64; 3])> {
        self.host.text_anchor(handle)
    }

    fn annotation(&self, handle: ItemHandle) -> Option<Annotation> {
        self.host.annotation(handle)
    }

    fn inserted_attribute(&self, attdef: ItemHandle, insert: ItemHandle) -> Option<Annotation> {
        self.host.inserted_attribute(attdef, insert)
    }

    fn entities<'a>(
        &'a self,
        scope: &EntityScope,
    ) -> Result<Box<dyn EntityCursor + 'a>, ProviderError> {
        let r = self.route(scope)?;
        let inner = r.provider.entities(&r.scope)?;
        Ok(Box::new(Prefixed {
            inner,
            key: r.key,
            stamp: r.stamp,
        }))
    }
}
