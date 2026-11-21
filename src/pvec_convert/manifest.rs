//! The CAD manifest and the one visibility resolver (#1626, EPIC C of iasbuilt/webapp #1434).
//!
//! [`cad_manifest`] lists a drawing (layers, layouts, viewports, named layer states) without
//! converting anything, [`CadChoices`] is what the user picks from that list, and
//! [`resolve_visibility`] is the single place that turns the two into a visible or hidden answer
//! per layer (spec section 1). The types serialise as `camelCase` JSON so a server can store and
//! relay them without a second model. The choices refuse unknown fields; the manifest does not.
//!
//! The listing and the resolver are stubs in this commit: the tests in
//! `tests/pvec_1626_cad_manifest.rs` are red until they are filled in.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::cad::CadSource;
use crate::pvec::DwgProvider;
use crate::pvec_convert::LayoutSetError;

/// What the user forces for one layer: wins over every flag, Defpoints included.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum LayerOverride {
    /// Draw the layer.
    Visible,
    /// Leave the layer out.
    Hidden,
}

/// The two states a resolved layer is in for model space, or for one layout or viewport.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Visibility {
    /// Drawn.
    Visible,
    /// Left out.
    Hidden,
}

/// A layer's value across the plots: [`Visibility`] plus "visible, but frozen in some viewport".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PlotsVisibility {
    /// Drawn in every plot.
    Visible,
    /// Left out of every plot.
    Hidden,
    /// Drawn, except through the viewports that freeze it.
    Viewport,
}

/// `show` or `hide`, for the file-level text switches (spec section 12).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TextSwitch {
    /// Notes on such layers are drawn.
    Show,
    /// Notes on such layers are left out.
    Hide,
}

/// The two text switches: notes on frozen layers, notes on no-plot layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CadText {
    /// Notes on frozen layers (default: show).
    pub frozen_layers: TextSwitch,
    /// Notes on no-plot layers (default: show in model mode, hide in plots mode).
    pub no_plot_layers: TextSwitch,
}

/// Model space or the print layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CadMode {
    /// One plan from model space.
    Model,
    /// One plan per print layout.
    Layouts,
}

/// What the user chose. Only the differences from the file are sent; absent means the file's value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CadChoices {
    /// Schema version, 1.
    pub schema: u32,
    /// Model space or print layouts.
    pub mode: CadMode,
    /// A named layer state applied under its mask before the overrides.
    #[serde(default)]
    pub layer_state: Option<String>,
    /// Hide every layer the chosen state does not mention.
    #[serde(default)]
    pub turn_off_layers_not_in_state: bool,
    /// The text switches, `None` for the defaults.
    #[serde(default)]
    pub text: Option<CadText>,
    /// Per-layer overrides, keyed by layer name (ASCII case-insensitive).
    #[serde(default)]
    pub layers: BTreeMap<String, LayerOverride>,
    /// Reserved for per-viewport overrides (`"LAYOUT/VP": { layer: override }`).
    #[serde(default)]
    pub viewports: BTreeMap<String, BTreeMap<String, LayerOverride>>,
    /// The per-layout plot choices, passed through untyped until the plot settings land.
    #[serde(default)]
    pub layouts: Vec<serde_json::Value>,
    /// The plot defaults, passed through untyped until the plot settings land.
    #[serde(default)]
    pub plot_defaults: Option<serde_json::Value>,
    /// The drawing unit, `None` for the server's rule.
    #[serde(default)]
    pub unit: Option<String>,
}

/// A layer's colour: an AutoCAD index or a true colour.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManifestColour {
    /// The AutoCAD Color Index, when the colour is one.
    pub aci: Option<u8>,
    /// The true colour, when it is one.
    pub rgb: Option<[u8; 3]>,
}

/// How many entities of a layer sit where.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LayerCounts {
    /// In model space.
    pub model: u64,
    /// In paper space, all layouts.
    pub paper: u64,
    /// Text entities (TEXT, MTEXT, ATTRIB), model and paper.
    pub text: u64,
}

/// A viewport that freezes a layer.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct VpRef {
    /// The layout's name.
    pub layout: String,
    /// The viewport's handle.
    pub viewport: String,
}

/// "Frozen in `frozen_in` of `of` viewports" for a [`PlotsVisibility::Viewport`] layer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlotsCount {
    /// Viewports of included layouts that freeze the layer.
    pub frozen_in: u32,
    /// Viewports of included layouts.
    pub of: u32,
}

/// What [`resolve_visibility`] decided for one layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ResolvedLayer {
    /// The layer's name, as the drawing spells it.
    pub name: String,
    /// In model space.
    pub model: Visibility,
    /// Across the plots.
    pub plots: PlotsVisibility,
    /// The count behind a [`PlotsVisibility::Viewport`].
    pub plots_count: PlotsCount,
    /// Layouts where the value differs from `plots`, by layout name.
    pub per_layout: BTreeMap<String, PlotsVisibility>,
    /// The answer through each viewport, keyed `layout/viewport`.
    pub per_viewport: BTreeMap<String, Visibility>,
}

/// One layer of the drawing, every flag.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManifestLayer {
    /// The name.
    pub name: String,
    /// The handle, when the drawing gives one.
    pub handle: Option<String>,
    /// Switched on.
    pub on: bool,
    /// Frozen.
    pub frozen: bool,
    /// Locked.
    pub locked: bool,
    /// Plots.
    pub plot: bool,
    /// Frozen in viewports made later.
    pub frozen_in_new_viewports: bool,
    /// The xref the layer belongs to (`xref|layer`).
    pub xref: Option<String>,
    /// Entities use the layer but the layer table has no row for it.
    pub missing_from_table: bool,
    /// The colour.
    pub color: Option<ManifestColour>,
    /// The linetype name.
    pub linetype: String,
    /// The lineweight in millimetres.
    pub lineweight_mm: Option<f64>,
    /// The transparency alpha.
    pub transparency: Option<u8>,
    /// The description.
    pub description: String,
    /// Entity counts.
    pub entity_count: LayerCounts,
    /// The viewports that freeze it.
    pub vp_frozen_in: Vec<VpRef>,
    /// What the resolver decided, filled by [`resolve_visibility`]'s caller.
    pub resolved: Option<ResolvedLayer>,
}

/// One viewport of a layout.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManifestViewport {
    /// The handle.
    pub id: String,
    /// The viewport number.
    pub index: i16,
    /// Its On flag.
    pub on: bool,
    /// Hidden when plotting.
    pub hide_plot: bool,
    /// Its own layer.
    pub layer: String,
    /// The layers frozen in this viewport only.
    pub frozen_layers: Vec<String>,
    /// `drawn`, `dropped` or `refused`, from the record alone.
    pub status: String,
    /// Why it is not drawn.
    pub reason: Option<String>,
}

/// One layout, skipped ones included.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManifestLayout {
    /// The name.
    pub name: String,
    /// The 1-based tab position among the paper layouts.
    pub tab: u32,
    /// The handle.
    pub handle: Option<String>,
    /// [`paper_sheets`](crate::pvec_convert::paper_sheets)'s answer.
    pub sheet: bool,
    /// `empty` or `emptyAfterView` when it is not a sheet.
    pub skip: Option<String>,
    /// The viewports.
    pub viewports: Vec<ManifestViewport>,
    /// Entities in the layout's paper space.
    pub paper_entity_count: u64,
}

/// What a named layer state restores (AutoCAD saves a mask with each state).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManifestMask {
    /// On or off.
    pub on: bool,
    /// Frozen.
    pub frozen: bool,
    /// Locked.
    pub locked: bool,
    /// Plot.
    pub plot: bool,
    /// Frozen in new viewports.
    pub new_vp_frozen: bool,
    /// Colour.
    pub color: bool,
    /// Linetype.
    pub linetype: bool,
    /// Lineweight.
    pub lineweight: bool,
    /// Plot style.
    pub plot_style: bool,
    /// Transparency.
    pub transparency: bool,
    /// Viewport settings.
    pub viewport: bool,
}

/// One layer as a named state recorded it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StateLayer {
    /// The layer's name.
    pub name: String,
    /// On.
    pub on: bool,
    /// Frozen.
    pub frozen: bool,
    /// Locked.
    pub locked: bool,
    /// Plots.
    pub plot: bool,
    /// Frozen in a viewport.
    pub vp_frozen: bool,
}

/// A named layer state, as the manifest lists it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManifestLayerState {
    /// The name.
    pub name: String,
    /// The description.
    pub description: String,
    /// What it restores.
    pub mask: ManifestMask,
    /// The layers it records.
    pub layers: Vec<StateLayer>,
    /// Names the state records that the drawing no longer has.
    pub missing_layers: Vec<String>,
    /// Drawing layers the state does not mention.
    pub unlisted_layers: Vec<String>,
}

/// A thing worth telling the user about the drawing.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManifestWarning {
    /// A stable code, `DUPLICATE_LAYER_NAME_CASE` for one.
    pub code: String,
    /// The detail.
    pub detail: String,
}

/// Model space in numbers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManifestModelSpace {
    /// Entities in model space.
    pub entity_count: u64,
    /// Of those, text.
    pub text_count: u64,
}

/// A drawing, listed without converting it (spec section 2.2, minus what a server adds: upload id,
/// expiry and file facts).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct CadManifest {
    /// Schema version, 1.
    pub schema: u32,
    /// Model space.
    pub model_space: ManifestModelSpace,
    /// Every layer.
    pub layers: Vec<ManifestLayer>,
    /// Every paper layout, in tab order.
    pub layouts: Vec<ManifestLayout>,
    /// The named layer states, in the drawing's order.
    pub layer_states: Vec<ManifestLayerState>,
    /// Warnings.
    pub warnings: Vec<ManifestWarning>,
}

/// Lists the drawing: layers with every flag, layouts in tab order (skipped ones included), their
/// viewports with frozen lists, and the named layer states. Opens the drawing once and converts
/// nothing.
///
/// # Errors
///
/// [`LayoutSetError::Open`] when the drawing does not open, [`LayoutSetError::Layouts`] when what
/// it lists cannot be read.
pub fn cad_manifest<P: DwgProvider>(source: CadSource<'_>) -> Result<CadManifest, LayoutSetError> {
    let _ = source;
    todo!("#1626: list layers, layouts, viewports and layer states in one pass")
}

/// One resolved answer per manifest layer: file flags, then the chosen state under its mask, then
/// the overrides (spec section 1.2). The only place that applies the visibility truth table.
#[must_use]
pub fn resolve_visibility(manifest: &CadManifest, choices: &CadChoices) -> Vec<ResolvedLayer> {
    let _ = (manifest, choices);
    todo!("#1626: apply the truth table for model space, plots and viewports")
}
