//! The CAD manifest and the one visibility resolver (#1626, EPIC C of iasbuilt/webapp #1434).
//!
//! [`cad_manifest`] lists a drawing (layers, layouts, viewports, named layer states) without
//! converting anything, [`CadChoices`] is what the user picks from that list, and
//! [`resolve_visibility`] is the single place that turns the two into a visible or hidden answer
//! per layer (spec section 1). The types serialise as `camelCase` JSON so a server can store and
//! relay them without a second model. The choices refuse unknown fields; the manifest does not.
//!

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::layouts::{listed_viewports, sheet_skip};
use super::{ViewportClass, viewport_class};
use crate::cad::{CadSource, ItemHandle};
use crate::pvec::{
    DwgProvider, EntityScope, LayerOverride, LayerStateRule, LayoutId, LayoutKind, ProviderError,
    SourceColour, SourceFact, ViewProfile,
};
use crate::pvec_convert::{LayoutSetError, SheetSkip};

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
    /// `paper` (the sheet's own view), `drawn`, `dropped` or `refused`, from the record alone.
    pub status: String,
    /// Why it is not drawn.
    pub reason: Option<String>,
}

/// A layout's paper, in millimetres whatever unit the drawing plots in.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManifestPaper {
    /// The size's name (`Tabloid`), when the drawing names one.
    pub size_name: Option<String>,
    /// Width in millimetres.
    pub width_mm: f64,
    /// Height in millimetres.
    pub height_mm: f64,
    /// `landscape` or `portrait`.
    pub orientation: String,
    /// The unit the drawing used (`mm` or `inches`), so a UI can show it back.
    pub unit: String,
    /// The plot rotation in degrees.
    pub rotation: u16,
}

/// What a layout plots: enough to ask for a thumbnail of the right area.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManifestPlot {
    /// The area: `display`, `extents`, `limits`, `view`, `window` or `layout`.
    pub plot_area: String,
    /// The window `[x0, y0, x1, y1]` when the area is `window`.
    pub window: Option<[f64; 4]>,
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
    /// The paper, `None` for a layout that is not a sheet. With `name`, `tab`, `plot` and
    /// `viewports` this is what a server needs to render one layout's thumbnail later.
    pub paper: Option<ManifestPaper>,
    /// The plot area, `None` for a layout that is not a sheet.
    pub plot: Option<ManifestPlot>,
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
    /// AutoCAD hides the state from its own list (a state it saved for itself).
    pub hidden: bool,
    /// The layers it records.
    pub layers: Vec<StateLayer>,
    /// Names the state records that the drawing no longer has.
    pub missing_layers: Vec<String>,
    /// Handles (hex) of recorded layers that have no name, because the layer was deleted from the
    /// table and the file keeps only its handle. `missing_layers` never holds a blank.
    pub missing_layer_handles: Vec<String>,
    /// Drawing layers the state does not mention.
    pub unlisted_layers: Vec<String>,
}

/// A thing worth telling the user about the drawing.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ManifestWarning {
    /// A stable code, `XREF_LAYERS` for one.
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

/// A layer's name as the case-insensitive key every lookup uses.
fn same(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b)
}

/// The hex handle the drawing shows for a viewport.
fn hex(h: Option<ItemHandle>) -> Option<String> {
    h.map(|h| format!("{:X}", h.get()))
}

/// Lists the drawing: layers with every flag, layouts in tab order (skipped ones included), their
/// viewports with frozen lists, and the named layer states. Opens the drawing once and converts
/// nothing. A layer is counted by the entity's own layer, not the effective one inside an INSERT.
///
/// # Errors
///
/// [`LayoutSetError::Open`] when the drawing does not open, [`LayoutSetError::Layouts`] when what
/// it lists cannot be read.
pub fn cad_manifest<P: DwgProvider>(source: CadSource<'_>) -> Result<CadManifest, LayoutSetError> {
    let provider = P::open(source).map_err(LayoutSetError::Open)?;
    let read = LayoutSetError::Layouts;
    let layouts = provider.layouts().map_err(read)?;
    let viewports = listed_viewports(&provider);

    let mut layers: Vec<ManifestLayer> = provider
        .layers()
        .map_err(read)?
        .into_iter()
        .map(|l| ManifestLayer {
            xref: l.name.split_once('|').map(|(x, _)| x.to_owned()),
            handle: hex(l.handle),
            on: !l.off,
            frozen: l.frozen,
            locked: l.locked,
            plot: l.plottable,
            frozen_in_new_viewports: l.frozen_in_new_viewports,
            color: l.color.and_then(|c| match c {
                SourceColour::Aci(n) => Some(ManifestColour {
                    aci: Some(n),
                    rgb: None,
                }),
                SourceColour::True { r, g, b } => Some(ManifestColour {
                    aci: None,
                    rgb: Some([r, g, b]),
                }),
                _ => None,
            }),
            linetype: l.linetype,
            lineweight_mm: l.lineweight.map(|w| f64::from(w) / 100.0),
            transparency: l.transparency,
            description: l.description,
            name: l.name,
            ..ManifestLayer::default()
        })
        .collect();

    let mut warnings = Vec::new();
    let xrefs = layers.iter().filter(|l| l.xref.is_some()).count();
    if xrefs > 0 {
        warnings.push(ManifestWarning {
            code: "XREF_LAYERS".into(),
            detail: format!("{xrefs} layers belong to xrefs that were not uploaded"),
        });
    }

    // One pass over every layout's entities for the per-layer counts; a layer the entities use
    // and the table lacks gets a row of its own.
    let tally = |layers: &mut Vec<ManifestLayer>,
                 scope: LayoutId,
                 model: bool|
     -> Result<(u64, u64), LayoutSetError> {
        let (mut all, mut text) = (0u64, 0u64);
        let mut cursor = provider
            .entities(&EntityScope::Layout(scope))
            .map_err(read)?;
        while let Some(e) = cursor.next_entity().map_err(read)? {
            let class = e.provenance.class_name.as_str();
            if class == "VIEWPORT" && !model {
                continue;
            }
            let at = match layers
                .iter()
                .position(|l| same(&l.name, &e.provenance.layer))
            {
                Some(i) => i,
                None => {
                    layers.push(ManifestLayer {
                        name: e.provenance.layer.clone(),
                        on: true,
                        plot: true,
                        linetype: "Continuous".into(),
                        missing_from_table: true,
                        ..ManifestLayer::default()
                    });
                    layers.len() - 1
                }
            };
            let is_text = matches!(class, "TEXT" | "MTEXT" | "ATTRIB" | "ATTDEF");
            let counts = &mut layers[at].entity_count;
            if model {
                counts.model += 1;
            } else {
                counts.paper += 1;
            }
            counts.text += u64::from(is_text);
            all += 1;
            text += u64::from(is_text);
        }
        Ok((all, text))
    };

    let mut model_space = ManifestModelSpace::default();
    if let Some(m) = layouts.iter().find(|l| l.kind == LayoutKind::Model) {
        let (all, text) = tally(&mut layers, m.id, true)?;
        model_space = ManifestModelSpace {
            entity_count: all,
            text_count: text,
        };
    }

    let mut rows = Vec::new();
    for (n, layout) in layouts
        .iter()
        .filter(|l| l.kind == LayoutKind::Paper)
        .enumerate()
    {
        let skip = sheet_skip(&provider, layout, &viewports).map_err(read)?;
        let sheet = skip.is_none();
        let (paper_entity_count, _) = tally(&mut layers, layout.id, false)?;
        let plot = match provider.layout_plot(layout.id) {
            Ok(SourceFact::Present(p)) if sheet => Some(p),
            _ => None,
        };
        let mut vps = Vec::new();
        for v in viewports
            .as_ref()
            .map_err(|e| read(e.clone()))?
            .iter()
            .filter(|v| v.layout == layout.id)
        {
            let (status, reason) = match viewport_class(v) {
                ViewportClass::Paper => ("paper", None),
                ViewportClass::Dropped(r) => ("dropped", Some(r)),
                ViewportClass::Refused(r) => ("refused", Some(r)),
                ViewportClass::Drawn(_) => ("drawn", None),
            };
            let id = hex(Some(v.handle)).unwrap_or_default();
            for name in &v.frozen_layers {
                if let Some(l) = layers.iter_mut().find(|l| same(&l.name, name)) {
                    l.vp_frozen_in.push(VpRef {
                        layout: layout.name.clone(),
                        viewport: id.clone(),
                    });
                }
            }
            vps.push(ManifestViewport {
                id,
                index: v.id,
                on: v.visible,
                hide_plot: v.hide_plot,
                layer: v.layer.clone(),
                frozen_layers: v.frozen_layers.clone(),
                status: status.into(),
                reason: reason.map(str::to_owned),
            });
        }
        rows.push(ManifestLayout {
            name: layout.name.clone(),
            tab: n as u32 + 1,
            handle: hex(None),
            sheet,
            skip: skip.map(|s| match s {
                SheetSkip::Empty => "empty".to_owned(),
                SheetSkip::EmptyAfterView => "emptyAfterView".to_owned(),
                _ => "failed".to_owned(),
            }),
            paper: plot.as_ref().map(|p| ManifestPaper {
                size_name: Some(p.paper_size.clone()).filter(|s| !s.is_empty()),
                width_mm: p.paper_width,
                height_mm: p.paper_height,
                orientation: if p.paper_width >= p.paper_height {
                    "landscape"
                } else {
                    "portrait"
                }
                .into(),
                unit: if p.paper_units == 0 { "inches" } else { "mm" }.into(),
                rotation: u16::try_from(p.rotation.rem_euclid(4) * 90).unwrap_or(0),
            }),
            plot: plot.as_ref().map(|p| ManifestPlot {
                plot_area: match p.plot_type {
                    0 => "display",
                    1 => "extents",
                    2 => "limits",
                    3 => "view",
                    4 => "window",
                    _ => "layout",
                }
                .into(),
                window: (p.plot_type == 4).then_some(p.window),
            }),
            viewports: vps,
            paper_entity_count,
        });
    }

    let states = match provider.layer_states() {
        Ok(s) => s,
        Err(ProviderError::CannotReport(_)) => Vec::new(),
        Err(e) => return Err(read(e)),
    };
    let layer_states = states
        .into_iter()
        .map(|s| ManifestLayerState {
            missing_layers: s
                .layers
                .iter()
                .filter(|sl| !sl.name.is_empty())
                .filter(|sl| !layers.iter().any(|l| same(&l.name, &sl.name)))
                .map(|sl| sl.name.clone())
                .collect(),
            missing_layer_handles: s
                .layers
                .iter()
                .filter(|sl| sl.name.is_empty())
                .filter_map(|sl| hex(sl.handle))
                .collect(),
            unlisted_layers: layers
                .iter()
                .filter(|l| !s.layers.iter().any(|sl| same(&l.name, &sl.name)))
                .map(|l| l.name.clone())
                .collect(),
            mask: ManifestMask {
                on: s.mask.on,
                frozen: s.mask.frozen,
                locked: s.mask.locked,
                plot: s.mask.plot,
                new_vp_frozen: s.mask.new_viewport_frozen,
                color: s.mask.color,
                linetype: s.mask.linetype,
                lineweight: s.mask.lineweight,
                plot_style: s.mask.plot_style,
                transparency: s.mask.transparency,
                viewport: s.mask.viewport,
            },
            layers: s
                .layers
                .iter()
                .filter(|l| !l.name.is_empty())
                .map(|l| StateLayer {
                    name: l.name.clone(),
                    on: !l.off,
                    frozen: l.frozen,
                    locked: l.locked,
                    plot: l.plottable,
                    vp_frozen: false,
                })
                .collect(),
            hidden: s.hidden,
            name: s.name,
            description: s.description,
        })
        .collect();

    Ok(CadManifest {
        schema: 1,
        model_space,
        layers,
        layouts: rows,
        layer_states,
        warnings,
    })
}

/// One resolved answer per manifest layer: file flags, then the chosen state under its mask, then
/// the overrides (spec section 1.2). The only place that applies the visibility truth table.
#[must_use]
pub fn resolve_visibility(manifest: &CadManifest, choices: &CadChoices) -> Vec<ResolvedLayer> {
    let state = choices
        .layer_state
        .as_deref()
        .and_then(|n| manifest.layer_states.iter().find(|s| same(&s.name, n)));
    let vis = |b: bool| {
        if b {
            Visibility::Visible
        } else {
            Visibility::Hidden
        }
    };
    let sheets = || manifest.layouts.iter().filter(|l| l.sheet);
    let windows = |l: &ManifestLayout| {
        l.viewports
            .iter()
            .filter(|v| v.status != "paper")
            .cloned()
            .collect::<Vec<_>>()
    };

    manifest
        .layers
        .iter()
        .map(|layer| {
            let recorded = state.and_then(|s| {
                s.layers
                    .iter()
                    .find(|sl| same(&sl.name, &layer.name))
                    .map(|sl| (s.mask, sl))
            });
            let pick =
                |mask: fn(&ManifestMask) -> bool, file: bool, rec: fn(&StateLayer) -> bool| {
                    recorded
                        .filter(|(m, _)| mask(m))
                        .map_or(file, |(_, sl)| rec(sl))
                };
            let mut on = pick(|m| m.on, layer.on, |s| s.on);
            let frozen = pick(|m| m.frozen, layer.frozen, |s| s.frozen);
            let plot = pick(|m| m.plot, layer.plot, |s| s.plot);
            if state.is_some() && recorded.is_none() && choices.turn_off_layers_not_in_state {
                on = false;
            }
            let forced = choices
                .layers
                .iter()
                .find(|(k, _)| same(k, &layer.name))
                .map(|(_, o)| *o);
            let forced = forced.map(|o| vis(o == LayerOverride::Visible));
            let base = on && !frozen && plot && !same(&layer.name, "DEFPOINTS");

            let mut plots_count = PlotsCount::default();
            let mut per_viewport = BTreeMap::new();
            let mut layout_frozen = Vec::new();
            for l in sheets() {
                let mut here = false;
                for v in windows(l) {
                    let frozen_here = v.frozen_layers.iter().any(|n| same(n, &layer.name));
                    plots_count.of += 1;
                    plots_count.frozen_in += u32::from(frozen_here);
                    here |= frozen_here;
                    per_viewport.insert(
                        format!("{}/{}", l.name, v.id),
                        forced.unwrap_or_else(|| vis(base && !frozen_here)),
                    );
                }
                layout_frozen.push((l.name.clone(), here));
            }
            let plots = match forced {
                Some(Visibility::Visible) => PlotsVisibility::Visible,
                Some(Visibility::Hidden) => PlotsVisibility::Hidden,
                None if !base => PlotsVisibility::Hidden,
                None if plots_count.frozen_in > 0 => PlotsVisibility::Viewport,
                None => PlotsVisibility::Visible,
            };
            let per_layout = layout_frozen
                .into_iter()
                .filter_map(|(name, here)| {
                    let v = match (forced.is_some() || !base, here) {
                        (true, _) => plots,
                        (false, true) => PlotsVisibility::Viewport,
                        (false, false) => PlotsVisibility::Visible,
                    };
                    (v != plots).then_some((name, v))
                })
                .collect();
            ResolvedLayer {
                name: layer.name.clone(),
                model: forced.unwrap_or_else(|| vis(on && !frozen)),
                plots,
                plots_count,
                per_layout,
                per_viewport,
            }
        })
        .collect()
}

/// The profile the server converts a plan with: `profile`, then the text switches from
/// `choices.text`, then the layer overrides and the per-viewport overrides from `choices` (spec
/// section 2.4). With nothing chosen the profile comes back as it went in.
#[must_use]
pub fn apply_choices(
    profile: ViewProfile,
    manifest: &CadManifest,
    choices: &CadChoices,
) -> ViewProfile {
    // A chosen layer state folds into the overrides where it changes the answer for the mode of
    // `choices` (model space, or the drawing-wide plot value in layouts mode), by resolving the
    // choices with and without the state and the user's own rows. A user's entry for the same
    // name wins, and a state's Visible never thaws a viewport: those entries stay apart from the
    // user's. Known gap: a text-bearing layer a state freezes loses its text even under
    // `frozenLayers: show`, because the fold is a Hidden.
    let mut profile = profile;
    let switch = |s: TextSwitch| match s {
        TextSwitch::Show => LayerStateRule::Ignore,
        TextSwitch::Hide => LayerStateRule::Exclude,
    };
    if let Some(t) = choices.text {
        profile.layers.frozen_text = switch(t.frozen_layers);
        profile.layers.no_plot_text = switch(t.no_plot_layers);
    }
    profile.layers.overrides = choices
        .layers
        .iter()
        .map(|(n, o)| (n.clone(), *o))
        .collect();
    if choices.layer_state.is_some() {
        let mut file = choices.clone();
        file.layer_state = None;
        file.turn_off_layers_not_in_state = false;
        file.layers.clear();
        let seen = |r: &ResolvedLayer| match choices.mode {
            CadMode::Model => r.model == Visibility::Visible,
            CadMode::Layouts => r.plots != PlotsVisibility::Hidden,
        };
        let mut with_state = file.clone();
        with_state.layer_state = choices.layer_state.clone();
        with_state.turn_off_layers_not_in_state = choices.turn_off_layers_not_in_state;
        profile.layers.state_overrides = resolve_visibility(manifest, &with_state)
            .iter()
            .zip(resolve_visibility(manifest, &file))
            .filter(|(s, f)| seen(s) != seen(f))
            .filter(|(s, _)| !choices.layers.keys().any(|k| same(k, &s.name)))
            .map(|(s, _)| {
                let o = if seen(s) {
                    LayerOverride::Visible
                } else {
                    LayerOverride::Hidden
                };
                (s.name.clone(), o)
            })
            .collect();
    }
    profile.layers.viewport_overrides = choices
        .viewports
        .iter()
        .flat_map(|(key, layers)| {
            layers
                .iter()
                .map(move |(n, o)| (key.clone(), n.clone(), *o))
        })
        .collect();
    profile
}
