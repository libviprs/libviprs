//! PDF optional content for the vector path and the wizard (issue #1637, C5.1).
//!
//! [`pdf_manifest`] lists the optional content groups of a PDF and how the document's default
//! configuration resolves each of them for viewing and for printing. [`apply_pdf_layers`]
//! rewrites that default configuration from a mode and a set of overrides, so the vector
//! conversion and the pdfium raster render both read the same answer.

use std::collections::{BTreeMap, BTreeSet};

use lopdf::content::Content;
use lopdf::{Dictionary, Document, IncrementalDocument, Object, ObjectId};
use thiserror::Error;

use crate::pdf::{page_rotation_of, page_visible_box};

/// One page of the document.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "camelCase")
)]
pub struct PdfPageInfo {
    /// The 1-based page number.
    pub page: usize,
    /// Displayed width after `/Rotate`, in points.
    pub width: f64,
    /// Displayed height after `/Rotate`, in points.
    pub height: f64,
    /// `/Rotate` in degrees.
    pub rotation_degrees: i64,
}

/// What a group resolves to in each mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "camelCase")
)]
pub struct PdfResolved {
    /// Shown when viewing.
    pub view: bool,
    /// Shown when printing.
    pub print: bool,
}

/// One optional content group.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "camelCase")
)]
pub struct PdfLayer {
    /// The object number of the group.
    pub id: u32,
    /// `/Name`.
    pub name: String,
    /// `/Intent`, `View` when the group says none.
    pub intent: Vec<String>,
    /// On in the default configuration, before any usage.
    pub on: bool,
    /// `Usage.View.ViewState` is OFF and `/D /AS` lists the View event for this group.
    pub view_hidden: bool,
    /// `Usage.Print.PrintState` is OFF and `/D /AS` lists the Print event for this group.
    pub print_hidden: bool,
    /// The labels above the group in `/D /Order`, outermost first. `None` when the group is not
    /// in `/Order` at all.
    pub order: Option<Vec<String>>,
    /// Indices into [`PdfManifest::rb_groups`] of the radio groups the layer belongs to.
    pub rb_groups: Vec<usize>,
    /// `BDC /OC` spans, XObjects and annotations that name this group directly.
    pub spans: u32,
    /// The answer in each mode.
    pub resolved: PdfResolved,
}

/// One optional content membership dictionary that some content refers to.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "camelCase")
)]
pub struct PdfOcmd {
    /// The object number.
    pub id: u32,
    /// `AnyOn`, `AllOn`, `AnyOff`, `AllOff`, or `VE` when a visibility expression decides.
    pub policy: String,
    /// The names of the groups it lists, or the groups its expression mentions.
    pub groups: Vec<String>,
    /// Spans, XObjects and annotations that name this dictionary.
    pub spans: u32,
    /// The answer in each mode.
    pub resolved: PdfResolved,
}

/// What a PDF says about its optional content.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "camelCase")
)]
pub struct PdfManifest {
    /// The pages.
    pub pages: Vec<PdfPageInfo>,
    /// The groups, in `/OCGs` order. Empty when the PDF has no `/OCProperties`.
    pub layers: Vec<PdfLayer>,
    /// The membership dictionaries content refers to.
    pub ocmds: Vec<PdfOcmd>,
    /// `/D /RBGroups`, as the object numbers of each radio group's members.
    pub rb_groups: Vec<Vec<u32>>,
    /// The names of `/Configs`, listed and never applied.
    pub configs: Vec<String>,
}

/// View or print.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "camelCase")
)]
pub enum PdfLayerMode {
    /// What a viewer shows (`on && !viewHidden`).
    #[default]
    View,
    /// What a printer prints (`on && !printHidden`).
    Print,
}

/// A user's word for one group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "camelCase")
)]
pub enum PdfLayerOverride {
    /// Show it.
    Visible,
    /// Hide it.
    Hidden,
}

/// The mode and the differences from the file.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(rename_all = "camelCase", default)
)]
pub struct PdfLayerChoices {
    /// Which usage decides the groups that are not overridden.
    pub mode: PdfLayerMode,
    /// Group name to override. A name two groups share overrides both.
    pub layers: BTreeMap<String, PdfLayerOverride>,
}

/// Why a PDF could not be read or rewritten.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PdfLayersError {
    /// The bytes are not a PDF this can read.
    #[error("cannot read the PDF: {0}")]
    Read(String),
    /// The PDF is encrypted.
    #[error("the PDF is encrypted")]
    Encrypted,
    /// The rewritten PDF could not be written.
    #[error("cannot write the PDF: {0}")]
    Write(String),
}

/// Nesting the visibility expressions, `/Order` trees and form streams are followed to.
const MAX_DEPTH: usize = 16;

fn deref<'a>(doc: &'a Document, o: &'a Object) -> Option<&'a Object> {
    doc.dereference(o).ok().map(|(_, o)| o)
}

fn text(doc: &Document, o: &Object) -> Option<String> {
    lopdf::decode_text_string(deref(doc, o)?).ok()
}

/// The object numbers an array of references holds, in order.
fn refs(doc: &Document, o: Option<&Object>) -> Vec<ObjectId> {
    let Some(Object::Array(a)) = o.and_then(|o| deref(doc, o)) else {
        return Vec::new();
    };
    a.iter().filter_map(|o| o.as_reference().ok()).collect()
}

struct Group {
    id: ObjectId,
    name: String,
    intent: Vec<String>,
    view_off: bool,
    print_off: bool,
}

/// The optional content of one document, as `/OCProperties /D` states it.
pub(super) struct OcModel {
    groups: Vec<Group>,
    base_off: bool,
    on: BTreeSet<ObjectId>,
    off: BTreeSet<ObjectId>,
    as_view: BTreeSet<ObjectId>,
    as_print: BTreeSet<ObjectId>,
    order: BTreeMap<ObjectId, Vec<String>>,
    rb: Vec<Vec<ObjectId>>,
    configs: Vec<String>,
}

fn ocproperties(doc: &Document) -> Option<&Dictionary> {
    let cat = doc.catalog().ok()?;
    deref(doc, cat.get(b"OCProperties").ok()?)?.as_dict().ok()
}

impl OcModel {
    /// `None` when the document has no `/OCProperties`.
    pub(super) fn load(doc: &Document) -> Option<Self> {
        let props = ocproperties(doc)?;
        let state = |g: &Dictionary, usage: &[u8], key: &[u8]| {
            g.get(b"Usage")
                .ok()
                .and_then(|u| deref(doc, u))
                .and_then(|u| u.as_dict().ok())
                .and_then(|u| u.get(usage).ok())
                .and_then(|u| deref(doc, u))
                .and_then(|u| u.as_dict().ok())
                .and_then(|u| u.get(key).ok())
                .and_then(|v| deref(doc, v))
                .and_then(|v| v.as_name().ok())
                .is_some_and(|v| v == b"OFF")
        };
        let groups: Vec<Group> = refs(doc, props.get(b"OCGs").ok())
            .into_iter()
            .filter_map(|id| {
                let g = doc.get_dictionary(id).ok()?;
                let intent = match g.get(b"Intent").ok().and_then(|i| deref(doc, i)) {
                    Some(Object::Name(n)) => vec![String::from_utf8_lossy(n).into_owned()],
                    Some(Object::Array(a)) => a
                        .iter()
                        .filter_map(|n| deref(doc, n)?.as_name().ok())
                        .map(|n| String::from_utf8_lossy(n).into_owned())
                        .collect(),
                    _ => vec!["View".to_owned()],
                };
                Some(Group {
                    id,
                    name: g
                        .get(b"Name")
                        .ok()
                        .and_then(|n| text(doc, n))
                        .unwrap_or_default(),
                    intent,
                    view_off: state(g, b"View", b"ViewState"),
                    print_off: state(g, b"Print", b"PrintState"),
                })
            })
            .collect();

        let empty = Dictionary::new();
        let d = props
            .get(b"D")
            .ok()
            .and_then(|d| deref(doc, d))
            .and_then(|d| d.as_dict().ok())
            .unwrap_or(&empty);
        let base_off = d
            .get(b"BaseState")
            .ok()
            .and_then(|b| deref(doc, b))
            .and_then(|b| b.as_name().ok())
            .is_some_and(|b| b == b"OFF");
        let (mut as_view, mut as_print) = (BTreeSet::new(), BTreeSet::new());
        if let Some(Object::Array(entries)) = d.get(b"AS").ok().and_then(|a| deref(doc, a)) {
            for e in entries {
                let Some(e) = deref(doc, e).and_then(|e| e.as_dict().ok()) else {
                    continue;
                };
                let target = match e
                    .get(b"Event")
                    .ok()
                    .and_then(|v| deref(doc, v))
                    .and_then(|v| v.as_name().ok())
                {
                    Some(b"View") => &mut as_view,
                    Some(b"Print") => &mut as_print,
                    _ => continue,
                };
                target.extend(refs(doc, e.get(b"OCGs").ok()));
            }
        }
        let mut order = BTreeMap::new();
        if let Some(Object::Array(items)) = d.get(b"Order").ok().and_then(|o| deref(doc, o)) {
            walk_order(doc, items, &mut Vec::new(), &mut order, 0);
        }
        let rb = match d.get(b"RBGroups").ok().and_then(|o| deref(doc, o)) {
            Some(Object::Array(a)) => a.iter().map(|g| refs(doc, Some(g))).collect(),
            _ => Vec::new(),
        };
        let configs = match props.get(b"Configs").ok().and_then(|o| deref(doc, o)) {
            Some(Object::Array(a)) => a
                .iter()
                .filter_map(|c| deref(doc, c)?.as_dict().ok()?.get(b"Name").ok())
                .filter_map(|n| text(doc, n))
                .collect(),
            _ => Vec::new(),
        };
        Some(Self {
            groups,
            base_off,
            on: refs(doc, d.get(b"ON").ok()).into_iter().collect(),
            off: refs(doc, d.get(b"OFF").ok()).into_iter().collect(),
            as_view,
            as_print,
            order,
            rb,
            configs,
        })
    }

    /// On in the default configuration, before any usage.
    fn config_on(&self, id: ObjectId) -> bool {
        if self.base_off {
            self.on.contains(&id)
        } else {
            !self.off.contains(&id)
        }
    }

    fn usage_hidden(&self, g: &Group, mode: PdfLayerMode) -> bool {
        match mode {
            PdfLayerMode::View => g.view_off && self.as_view.contains(&g.id),
            PdfLayerMode::Print => g.print_off && self.as_print.contains(&g.id),
        }
    }

    fn group_visible(&self, g: &Group, mode: PdfLayerMode) -> bool {
        // Viewers leave a group whose intent is not View always on.
        !g.intent.iter().any(|i| i == "View" || i == "All")
            || (self.config_on(g.id) && !self.usage_hidden(g, mode))
    }

    fn state(&self, id: ObjectId, mode: PdfLayerMode) -> bool {
        match self.groups.iter().find(|g| g.id == id) {
            Some(g) => self.group_visible(g, mode),
            None => self.config_on(id),
        }
    }

    /// Whether content marked with `target` (an OCG or an OCMD, by reference) shows.
    pub(super) fn shows(&self, doc: &Document, target: &Object, mode: PdfLayerMode) -> bool {
        let Ok(id) = target.as_reference() else {
            return true;
        };
        let Ok(dict) = doc.get_dictionary(id) else {
            return true;
        };
        if dict.get(b"Type").ok().and_then(|t| t.as_name().ok()) != Some(&b"OCMD"[..]) {
            return self.state(id, mode);
        }
        if let Some(Object::Array(ve)) = dict.get(b"VE").ok().and_then(|v| deref(doc, v)) {
            return self.expression(doc, ve, mode, 0);
        }
        let members = match dict.get(b"OCGs").ok() {
            Some(Object::Reference(r)) if doc.get_dictionary(*r).is_ok() => vec![*r],
            other => refs(doc, other),
        };
        if members.is_empty() {
            return true;
        }
        let states = members.iter().map(|m| self.state(*m, mode));
        match dict
            .get(b"P")
            .ok()
            .and_then(|p| p.as_name().ok())
            .unwrap_or(b"AnyOn")
        {
            b"AllOn" => states.clone().all(|s| s),
            b"AnyOff" => states.clone().any(|s| !s),
            b"AllOff" => states.clone().all(|s| !s),
            _ => states.clone().any(|s| s),
        }
    }

    fn expression(&self, doc: &Document, ve: &[Object], mode: PdfLayerMode, depth: usize) -> bool {
        if depth > MAX_DEPTH {
            return true;
        }
        let Some(op) = ve
            .first()
            .and_then(|o| deref(doc, o))
            .and_then(|o| o.as_name().ok())
        else {
            return true;
        };
        let operand = |o: &Object| match o {
            Object::Reference(id) => match deref(doc, o) {
                Some(Object::Array(a)) => self.expression(doc, a, mode, depth + 1),
                _ => self.state(*id, mode),
            },
            Object::Array(a) => self.expression(doc, a, mode, depth + 1),
            _ => true,
        };
        let mut rest = ve[1..].iter().map(operand);
        match op {
            b"Not" => !rest.next().unwrap_or(false),
            b"Or" => rest.any(|s| s),
            _ => rest.all(|s| s),
        }
    }
}

/// `/Order`: an element is a group, or an array. An array that starts with a text string is a
/// labelled node over the rest of it; any other array hangs under the group before it.
fn walk_order(
    doc: &Document,
    items: &[Object],
    path: &mut Vec<String>,
    out: &mut BTreeMap<ObjectId, Vec<String>>,
    depth: usize,
) {
    if depth > MAX_DEPTH {
        return;
    }
    let mut previous: Option<String> = None;
    for item in items {
        match deref(doc, item) {
            Some(Object::Array(a)) => {
                let label = match a.first().and_then(|f| deref(doc, f)) {
                    Some(s @ Object::String(..)) => lopdf::decode_text_string(s).ok(),
                    _ => None,
                };
                let (label, rest) = match label {
                    Some(l) => (Some(l), &a[1..]),
                    None => (previous.clone(), &a[..]),
                };
                let pushed = label.is_some();
                path.extend(label);
                walk_order(doc, rest, path, out, depth + 1);
                if pushed {
                    path.pop();
                }
            }
            Some(Object::Dictionary(g)) => {
                if let Ok(id) = item.as_reference() {
                    out.entry(id).or_insert_with(|| path.clone());
                    previous = g.get(b"Name").ok().and_then(|n| text(doc, n));
                }
            }
            _ => {}
        }
    }
}

/// Marked-content spans, forms and annotations that name an OCG or an OCMD, by target, as
/// authored: a form shared by two pages is counted once.
fn count_spans(doc: &Document) -> BTreeMap<ObjectId, u32> {
    struct Counter<'a> {
        doc: &'a Document,
        counts: BTreeMap<ObjectId, u32>,
        seen: BTreeSet<ObjectId>,
    }
    impl Counter<'_> {
        fn bump(&mut self, o: Option<&Object>) {
            if let Some(Object::Reference(id)) = o {
                *self.counts.entry(*id).or_insert(0) += 1;
            }
        }

        fn stream(&mut self, bytes: &[u8], resources: Option<&Dictionary>, depth: usize) {
            let Ok(content) = Content::decode(bytes) else {
                return;
            };
            let doc = self.doc;
            for op in content.operations {
                match (op.operator.as_str(), op.operands.as_slice()) {
                    ("BDC", [Object::Name(tag), Object::Name(prop)]) if tag == b"OC" => {
                        let target = resources
                            .and_then(|r| r.get(b"Properties").ok())
                            .and_then(|p| deref(doc, p))
                            .and_then(|p| p.as_dict().ok())
                            .and_then(|p| p.get(prop).ok());
                        self.bump(target);
                    }
                    ("Do", [Object::Name(name)]) => {
                        let Some(Object::Reference(id)) = resources
                            .and_then(|r| r.get(b"XObject").ok())
                            .and_then(|x| deref(doc, x))
                            .and_then(|x| x.as_dict().ok())
                            .and_then(|x| x.get(name).ok())
                        else {
                            continue;
                        };
                        if depth >= MAX_DEPTH || !self.seen.insert(*id) {
                            continue;
                        }
                        let Ok(Object::Stream(form)) = doc.get_object(*id) else {
                            continue;
                        };
                        self.bump(form.dict.get(b"OC").ok());
                        let is_form = form
                            .dict
                            .get(b"Subtype")
                            .ok()
                            .and_then(|s| s.as_name().ok())
                            == Some(&b"Form"[..]);
                        if is_form {
                            let own = form
                                .dict
                                .get(b"Resources")
                                .ok()
                                .and_then(|r| deref(doc, r))
                                .and_then(|r| r.as_dict().ok());
                            let data = form
                                .decompressed_content()
                                .unwrap_or_else(|_| form.content.clone());
                            self.stream(&data, own.or(resources), depth + 1);
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    let mut c = Counter {
        doc,
        counts: BTreeMap::new(),
        seen: BTreeSet::new(),
    };
    for page_id in doc.get_pages().into_values() {
        let resources = doc.get_page_resources(page_id).ok().and_then(|(r, ids)| {
            r.or_else(|| ids.first().and_then(|i| doc.get_dictionary(*i).ok()))
        });
        if let Ok(bytes) = doc.get_page_content(page_id) {
            c.stream(&bytes, resources, 0);
        }
        if let Ok(annots) = doc.get_page_annotations(page_id) {
            for a in annots {
                c.bump(a.get(b"OC").ok());
            }
        }
    }
    c.counts
}

fn load(bytes: &[u8]) -> Result<Document, PdfLayersError> {
    let doc = Document::load_mem(bytes).map_err(|e| PdfLayersError::Read(e.to_string()))?;
    if doc.is_encrypted() {
        return Err(PdfLayersError::Encrypted);
    }
    Ok(doc)
}

/// List the pages and optional content groups of a PDF.
///
/// `layers` is empty for a PDF with no `/OCProperties`. Each group's `resolved` answer follows
/// the PDF truth table: on in `/D` (`/BaseState`, then `/ON` and `/OFF`), and not hidden by a
/// usage (`ViewState` or `PrintState` OFF) that `/D /AS` applies for the mode. A membership
/// dictionary is evaluated over the resolved groups.
///
/// # Errors
///
/// See [`PdfLayersError`].
pub fn pdf_manifest(bytes: &[u8]) -> Result<PdfManifest, PdfLayersError> {
    fn mentioned(doc: &Document, o: &Object, out: &mut Vec<ObjectId>, depth: usize) {
        if depth > MAX_DEPTH {
            return;
        }
        match o {
            Object::Reference(id) => match deref(doc, o) {
                Some(Object::Array(a)) => {
                    a.iter().for_each(|x| mentioned(doc, x, out, depth + 1));
                }
                _ => out.push(*id),
            },
            Object::Array(a) => a.iter().for_each(|x| mentioned(doc, x, out, depth + 1)),
            _ => {}
        }
    }
    let doc = &load(bytes)?;
    let pages = doc
        .get_pages()
        .into_iter()
        .map(|(n, id)| {
            let [l, b, r, t] = page_visible_box(doc, id).unwrap_or([0.0, 0.0, 612.0, 792.0]);
            let degrees = page_rotation_of(doc, id).map_or(0, crate::pdf::PageRotation::as_degrees);
            let (w, h) = (r - l, t - b);
            let (width, height) = if degrees % 180 == 0 { (w, h) } else { (h, w) };
            PdfPageInfo {
                page: n as usize,
                width,
                height,
                rotation_degrees: degrees,
            }
        })
        .collect();
    let Some(model) = OcModel::load(doc) else {
        return Ok(PdfManifest {
            pages,
            layers: Vec::new(),
            ocmds: Vec::new(),
            rb_groups: Vec::new(),
            configs: Vec::new(),
        });
    };
    let spans = count_spans(doc);
    let resolved = |show: &dyn Fn(PdfLayerMode) -> bool| PdfResolved {
        view: show(PdfLayerMode::View),
        print: show(PdfLayerMode::Print),
    };
    let layers = model
        .groups
        .iter()
        .map(|g| PdfLayer {
            id: g.id.0,
            name: g.name.clone(),
            intent: g.intent.clone(),
            on: model.config_on(g.id),
            view_hidden: model.usage_hidden(g, PdfLayerMode::View),
            print_hidden: model.usage_hidden(g, PdfLayerMode::Print),
            order: model.order.get(&g.id).cloned(),
            rb_groups: model
                .rb
                .iter()
                .enumerate()
                .filter(|(_, members)| members.contains(&g.id))
                .map(|(i, _)| i)
                .collect(),
            spans: spans.get(&g.id).copied().unwrap_or(0),
            resolved: resolved(&|m| model.group_visible(g, m)),
        })
        .collect();
    let ocmds = spans
        .iter()
        .filter_map(|(id, n)| {
            let d = doc.get_dictionary(*id).ok()?;
            (d.get(b"Type").ok()?.as_name().ok()? == b"OCMD").then(|| {
                let target = Object::Reference(*id);
                // The groups the OCMD lists, or its expression mentions.
                let mut members = Vec::new();
                if let Some(Object::Array(ve)) = d.get(b"VE").ok().and_then(|v| deref(doc, v)) {
                    ve.iter()
                        .skip(1)
                        .for_each(|x| mentioned(doc, x, &mut members, 0));
                } else {
                    match d.get(b"OCGs").ok() {
                        Some(Object::Reference(r)) if doc.get_dictionary(*r).is_ok() => {
                            members.push(*r);
                        }
                        other => members = refs(doc, other),
                    }
                }
                PdfOcmd {
                    id: id.0,
                    policy: if d.has(b"VE") {
                        "VE".to_owned()
                    } else {
                        d.get(b"P").ok().and_then(|p| p.as_name().ok()).map_or_else(
                            || "AnyOn".to_owned(),
                            |p| String::from_utf8_lossy(p).into_owned(),
                        )
                    },
                    groups: members
                        .into_iter()
                        .map(|g| {
                            model
                                .groups
                                .iter()
                                .find(|x| x.id == g)
                                .map_or_else(|| format!("{} {} R", g.0, g.1), |x| x.name.clone())
                        })
                        .collect(),
                    spans: *n,
                    resolved: resolved(&|m| model.shows(doc, &target, m)),
                }
            })
        })
        .collect();
    Ok(PdfManifest {
        pages,
        layers,
        ocmds,
        rb_groups: model
            .rb
            .iter()
            .map(|g| g.iter().map(|id| id.0).collect())
            .collect(),
        configs: model.configs,
    })
}

/// Rewrite the default configuration of a PDF from `choices`.
///
/// The mode picks which usage decides the groups that are not overridden, an override wins over
/// both, and the result is written into `/OCProperties /D` as `/BaseState /ON` with explicit
/// `/ON` and `/OFF` lists, with `/AS` removed so no usage can change the answer afterwards. A
/// group in a radio group that is forced visible turns the others of its radio groups off
/// unless they are overridden too. pdfium reads a group's own `ViewState` and `PrintState`
/// without asking `/AS`, so those two keys are removed from the groups that carry them.
///
/// The result is the input followed by an incremental update, so everything else is byte for
/// byte what it was. A PDF with no `/OCProperties` comes back unchanged.
///
/// # Errors
///
/// See [`PdfLayersError`].
pub fn apply_pdf_layers(
    bytes: &[u8],
    choices: &PdfLayerChoices,
) -> Result<Vec<u8>, PdfLayersError> {
    let doc = load(bytes)?;
    let Some(model) = OcModel::load(&doc) else {
        return Ok(bytes.to_vec());
    };
    let mut visible: Vec<bool> = model
        .groups
        .iter()
        .map(|g| match choices.layers.get(&g.name) {
            Some(PdfLayerOverride::Visible) => true,
            Some(PdfLayerOverride::Hidden) => false,
            None => model.group_visible(g, choices.mode),
        })
        .collect();
    for (i, g) in model.groups.iter().enumerate() {
        if choices.layers.get(&g.name) != Some(&PdfLayerOverride::Visible) {
            continue;
        }
        for sibling in model.rb.iter().filter(|rb| rb.contains(&g.id)).flatten() {
            if let Some(j) = model.groups.iter().position(|s| s.id == *sibling)
                && j != i
                && !choices.layers.contains_key(&model.groups[j].name)
            {
                visible[j] = false;
            }
        }
    }
    let list = |want: bool| -> Object {
        Object::Array(
            model
                .groups
                .iter()
                .zip(&visible)
                .filter(|(_, v)| **v == want)
                .map(|(g, _)| Object::Reference(g.id))
                .collect(),
        )
    };

    let mut inc = IncrementalDocument::create_from(bytes.to_vec(), doc);
    let doc = inc.get_prev_documents();
    let write_err = |e: lopdf::Error| PdfLayersError::Write(e.to_string());

    // Where /D lives: its own object, inside an /OCProperties object, or inside the catalog.
    let root = doc
        .trailer
        .get(b"Root")
        .and_then(Object::as_reference)
        .map_err(write_err)?;
    let props_slot = doc.catalog().map_err(write_err)?.get(b"OCProperties").ok();
    let (props_id, d_ref) = match props_slot {
        Some(Object::Reference(p)) => {
            let d_ref = doc
                .get_dictionary(*p)
                .ok()
                .and_then(|d| d.get(b"D").ok())
                .and_then(|d| d.as_reference().ok());
            (Some(*p), d_ref)
        }
        _ => (None, None),
    };
    let mut d = ocproperties(doc)
        .and_then(|p| p.get(b"D").ok())
        .and_then(|d| deref(doc, d))
        .and_then(|d| d.as_dict().ok())
        .cloned()
        .unwrap_or_default();
    d.set("BaseState", Object::Name(b"ON".to_vec()));
    d.set("ON", list(true));
    d.set("OFF", list(false));
    d.remove(b"AS");

    // `Usage` and its `View` and `Print` entries may each be direct or indirect, and the state
    // key is dropped wherever it lives.
    let mut strip: Vec<(ObjectId, Dictionary)> = Vec::new();
    for g in model.groups.iter().filter(|g| g.view_off || g.print_off) {
        let Ok(mut group) = doc.get_dictionary(g.id).cloned() else {
            continue;
        };
        let (usage_id, mut usage) = match group.get(b"Usage") {
            Ok(Object::Reference(u)) => match doc.get_dictionary(*u) {
                Ok(d) => (Some(*u), d.clone()),
                Err(_) => continue,
            },
            Ok(Object::Dictionary(d)) => (None, d.clone()),
            _ => continue,
        };
        for (event, key) in [(&b"View"[..], &b"ViewState"[..]), (b"Print", b"PrintState")] {
            match usage.get_mut(event) {
                Ok(Object::Dictionary(u)) => {
                    u.remove(key);
                }
                Ok(Object::Reference(r)) => {
                    if let Ok(mut u) = doc.get_dictionary(*r).cloned() {
                        u.remove(key);
                        strip.push((*r, u));
                    }
                }
                _ => {}
            }
        }
        if let Some(u) = usage_id {
            strip.push((u, usage));
        } else {
            group.set("Usage", usage);
            strip.push((g.id, group));
        }
    }

    match (props_id, d_ref) {
        (_, Some(d_id)) => inc.new_document.set_object(d_id, d),
        (Some(p), None) => {
            let mut props = inc
                .get_prev_documents()
                .get_dictionary(p)
                .map_err(write_err)?
                .clone();
            props.set("D", d);
            inc.new_document.set_object(p, props);
        }
        (None, None) => {
            let mut cat = inc
                .get_prev_documents()
                .catalog()
                .map_err(write_err)?
                .clone();
            if let Ok(Object::Dictionary(p)) = cat.get_mut(b"OCProperties") {
                p.set("D", d);
            }
            inc.new_document.set_object(root, cat);
        }
    }
    for (id, group) in strip {
        inc.new_document.set_object(id, group);
    }
    let mut out = Vec::with_capacity(bytes.len() + 1024);
    inc.save_to(&mut out)
        .map_err(|e| PdfLayersError::Write(e.to_string()))?;
    Ok(out)
}
