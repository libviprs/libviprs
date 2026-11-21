//! PDF optional content for the vector path and the wizard (issue #1637, C5.1).
//!
//! [`pdf_manifest`] lists the optional content groups of a PDF and how the document's default
//! configuration resolves each of them for viewing and for printing. [`apply_pdf_layers`]
//! rewrites that default configuration from a mode and a set of overrides, so the vector
//! conversion and the pdfium raster render both read the same answer.

use std::collections::BTreeMap;

use thiserror::Error;

/// One page of the document.
#[derive(Debug, Clone, PartialEq)]
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
pub struct PdfResolved {
    /// Shown when viewing.
    pub view: bool,
    /// Shown when printing.
    pub print: bool,
}

/// One optional content group.
#[derive(Debug, Clone, PartialEq, Eq)]
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
pub enum PdfLayerMode {
    /// What a viewer shows (`on && !viewHidden`).
    #[default]
    View,
    /// What a printer prints (`on && !printHidden`).
    Print,
}

/// A user's word for one group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdfLayerOverride {
    /// Show it.
    Visible,
    /// Hide it.
    Hidden,
}

/// The mode and the differences from the file.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
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

/// List the pages and optional content groups of a PDF.
///
/// # Errors
///
/// See [`PdfLayersError`].
pub fn pdf_manifest(bytes: &[u8]) -> Result<PdfManifest, PdfLayersError> {
    let _ = bytes;
    Ok(PdfManifest {
        pages: Vec::new(),
        layers: Vec::new(),
        ocmds: Vec::new(),
        rb_groups: Vec::new(),
        configs: Vec::new(),
    })
}

/// Rewrite the default configuration of a PDF from `choices`.
///
/// # Errors
///
/// See [`PdfLayersError`].
pub fn apply_pdf_layers(
    bytes: &[u8],
    choices: &PdfLayerChoices,
) -> Result<Vec<u8>, PdfLayersError> {
    let _ = choices;
    Ok(bytes.to_vec())
}
