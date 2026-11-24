# The conversion API

`libviprs::pvec_convert::PvecConversion` (feature `pvec-convert`, which joins `pvec-write` and `dwg-acadrust`) is the one surface an
application calls. It is built the way `EngineBuilder` is.

```rust
let out = PvecConversion::new(CadSource::Path(&dwg), &dest)
    .unit(SourceUnit::Meters)          // when the drawing declares none
    .profile(ConversionProfile::Default)
    .convert()?;
assert_eq!(out.report.totals.encountered, out.fidelity.encountered());
```

## Builder

`view_profile`, `limits`, `unit`, `leaves` (spatial directory and Tier 0), `cancel`, `observer`, `queue_capacity`, `calibration` (a `Calibration`, written into the artifact's calibration block),
`plot_settings`, `profile`, `allowed_capabilities`, `capability_policy`. `probe` returns the per-class counts without writing, `convert`
writes. `convert_with::<P>` and `probe_with::<P>` take any `DwgProvider`; the trait carries no unit, so `unit` is required there.

## Plot settings

`plot_settings(PlotSettings)` (#1628) chooses what a paper sheet plots, over the layout's own plot settings. `PlotSettings::default()` is the layout as it stands, which is what a conversion does without the call. All of it needs plot settings the drawing reports; without them the frame is the extents whatever is asked and the report notes `plot-settings-not-applied`.

| field | effect |
|---|---|
| `area` | `PlotArea::Layout` (default) frames the paper from its margins and origin. `PlotArea::Extents` places the paper at the lower left of the extents: the sheet's kept geometry plus the rectangles of its drawn viewports. |
| `paper_mm` | Paper size in millimetres, width then height, before the layout's rotation. It replaces the layout's paper and clears the layout's margins. A size that is zero, negative or not finite is refused as `paper-size-zero` and the frame is the extents. |
| `orientation` | `Orientation::Portrait` or `Landscape`: turns the paper, after the layout's rotation, to the shape asked for. `None` is the layout's own. |
| `scale` | A `PlotScale { paper, paper_unit, drawing }` in place of the layout's plot scale: `paper` paper units of `paper_unit` to `drawing` drawing units (the conversion's own unit), as the dialog's custom row and its presets give it. `1 inch = 50 units` is `{ 1, Inches, 50 }` on a sheet of any unit. It is the calibration record, with `paper_unit` as its paper unit, and the sheet's linework is drawn at it, as at the layout's own (#1665). |
| `offset_mm` | The plot offset, in place of the layout's own plot origin (the dialog shows one value). `None` is the layout's own. |
| `center` | Centres the extents on the printable area (the paper less its margins); the offset is then ignored. Plot area Extents only: under Layout it is ignored with `plot-centre-not-applicable`. |
| `fit_to_paper` | Scales what the plot covers into the printable area, the tighter side binding. Under Extents that is the extents. Under Layout it is the layout's own printable area, which only changes anything with a paper override (otherwise `plot-fit-no-change`). The frame is the window of the drawing the paper then shows, so the report's box is not the paper: read the paper size from the report's plot fields. The calibration stays the plot scale, and the fit ratio is the note `plot-fit-ratio 1:N`. |
| `print_lineweights` | Lineweights print. `Some(false)` drops widths. `None` is the layout's own. |
| `paperspace_last` | Paper space draws after what the viewports show. `None` is the layout's own. |

The sheet's linework, its viewport windows, its extents and a plot window are drawn at the plot scale, so the box and the unit are paper whatever the scale (#1665); a sheet at 1:1 is unchanged. The report's `sheet.notes` carry `plot-scale-applied`, `plot-area-extents`, `plot-offset`, `plot-centered`, `plot-centre-not-applicable`, `plot-paper-overridden`, `plot-orientation`, `plot-scale-overridden`, `plot-lineweights-overridden`, `plot-fit-to-paper`, `plot-fit-ratio 1:N`, `plot-fit-no-change` and `plot-fit-skipped`.

## Thumbnail

With `--features pvec-raster-thumbnail`, `PvecConversion::thumbnail(width_px)` also draws the converted sheet into `PvecConversionOutput::thumbnail` as an 8-bit RGB PNG. Strokes only, black and one pixel wide on opaque white; text is not drawn. The width is the caller's and the height follows the frame's aspect, so a long thin sheet gives a tall or a flat image. A paper layout is drawn edge to edge. Model space, and a layout without plot settings, leave 2 percent of the frame's longer side empty on every side. A paper sheet with nothing to stroke yields a blank picture and no artifact (`written == 0`). One call per layout: `PvecLayoutSet::sheets` passes it on to each sheet. A zero width is `PvecConvertError::Thumbnail`.

`pvec::thumbnail::thumbnail(artifact, width_px)` draws an artifact that is already published, framed on its declared bounds with no margin.

## Profiles and the capability gate

The caller says before the run which PVec capabilities it may use. An entity needing one outside the set is refused, never reshaped.

| profile | capabilities | entities | policy |
|---|---|---|---|
| `Strict` | those of the structurally exact rows | `SupportedExact` only | `RejectInput` |
| `Default` | every row of the mapping table | exact and equivalent | `RejectEntity` |
| `Permissive` | every row of the mapping table | exact and equivalent | `Downgrade` |

`RejectInput` aborts with `CapabilityNotAllowed` naming the capability and the entity. `RejectEntity` leaves the entity out and counts it as
`RequiresPVecCapability`, reason `capability-not-allowed`. `Downgrade` has no weaker representation to fall back to in v1 (none is not a
tessellation), so today it records `downgrade-unavailable` and omits the entity.

## Output and errors

`PvecConversionOutput` has the published artifact path, the report, the independent per-class counts, the entities written and the peak queue
depth. `PvecConvertError` keeps every lower error as `source()`: open, view, flatten, unit required, write, tally, report, publish, georeference
(a deprecated control point that is not finite), capability not allowed, cancelled. After any error the destination and its report are as they
were.

## Progress, cancellation, backpressure

The classifier and its consumer meet in a bounded queue (default 64, `queue_capacity`), the observer hears the stages as `EngineEvent`s, and the
cancel token is polled once per batch of 256 entities and during the write. A cancel that fires during a spool flush stops the write there (#1463).

## What it writes

The artifact (with its calibration block when there is a calibration, `docs/pvec-georeference.md`) and `<dest>.report.json`
(`docs/pvec-conversion-report.md`), published together or not at all. No georeference sidecar is written since #1554. Diagnostics have stable codes (`docs/pvec-diagnostics.md`).

## Listing a drawing before converting it

`pvec_convert::cad_manifest::<P>(source)` (#1626) opens the drawing once and returns a `CadManifest` (`camelCase` serde, `schema` 1). It converts and
writes nothing, so an import wizard can show the drawing first and ask second.

```rust
let manifest = cad_manifest::<AcadrustProvider>(CadSource::Path(&dwg))?;
let choices: CadChoices = serde_json::from_str(r#"{"schema":1,"mode":"layouts","layerState":"Plot"}"#)?;
let layers = resolve_visibility(&manifest, &choices);   // one ResolvedLayer per manifest layer
```

The manifest has `layers` (every flag, the layer's colour, linetype, lineweight, transparency, plot style and description, and counts per
layer), `layouts` in tab order with the skipped ones included (`ManifestPaper`, `ManifestPlot`, the layout's handle, and one `ManifestViewport`
per viewport with its frozen layers and whether the print draws, drops or refuses it), `layer_states` (`ManifestLayerState`, with the layers
a state names that the drawing no longer has under `missingLayers`/`missingLayerHandles`, and the ones it does not mention as unlisted) and
`warnings` (`XREF_LAYERS` is the only code). A layer is counted by the entity's own layer, not the effective one inside an INSERT.

`CadChoices` holds only the differences from the file: `mode` (`model` or `layouts`), `layer_state`, `turn_off_layers_not_in_state`, `text`,
`layers` (a `LayerOverride` per layer name, ASCII case-insensitive), and `unit`. It refuses unknown fields; the manifest does not.
`resolve_visibility` is the one place the truth table is applied: the file's flags, then the chosen state under its mask, then the overrides,
with Defpoints and "turn off layers not in the state" handled there. It needs `DwgProvider::layer_states()` and `layout_handle()`, which
default to "cannot report" (an empty list and `None`), so a provider that does not read them gets an empty `layer_states`.

`apply_choices(profile, &manifest, &choices)` (#1627) is how those choices reach a conversion: it returns the `ViewProfile` with the text
switches from `choices.text`, the layer overrides (`LayerVisibility::overrides`) and the per-viewport overrides applied, and the chosen
layer state folded in where it changes the answer. A user's entry for a layer wins over the state's, and a state's `visible` never thaws a viewport.
A layer a state only freezes goes in `state_frozen`, a frozen flag like the file's own, so it answers to `frozen` and `frozen_text` and its notes
follow `frozenLayers`. An entity a layer override hides is counted as `layers.overrides` (`Exclusion::LayerOverridden`, code `LAYER_HIDDEN`).

`check_layer_state(&manifest, &choices)` (#1675) says whether the layer state `choices` names is one the drawing stores (ASCII case-insensitive). When it isn't, it returns `LayerStateNotFound` (`code()` is `LAYER_STATE_NOT_FOUND`) with the name asked for and the states there are, without the ones AutoCAD hides. `apply_choices` itself still falls back to the file's table for an unknown name, so run the check first. In model mode `apply_choices` leaves `viewport_overrides` empty, since those overrides belong to sheets.

## Every sheet of a drawing

`PvecLayoutSet` (#1564) converts each paper layout to its own artifact, one call for the drawing. It opens the drawing once, walks the paper
layouts in tab order and runs the conversion above under `ViewProfile::print_profile(layout)` for each.

```rust
let set = PvecLayoutSet::new(CadSource::Path(&dwg), &out_dir)
    .layouts(["A-101", "A-102"])           // optional; an unknown name fails the whole set
    .sheets(|c| c.components(StreamSelection::ALL))
    .convert()?;
for s in set.failed() { eprintln!("{}: {:?}", s.layout, s.outcome); }
```

Each sheet is `<stem>.<NN>-<slug>.pvec` with its report beside it. `NN` is the layout's 1-based position among all the drawing's paper layouts,
zero padded to the width of their count and at least two digits; the slug is the name with every byte outside `[A-Za-z0-9._-]` turned into
`-`, runs collapsed, `-` and `.` trimmed, at most 64 bytes, `layout` when nothing is left. The stem is the path's own stem, and `.stem(..)` is
required for bytes. `.sheets(|c| ...)` takes `PvecConversion`'s builder; the profile's space is the layout whatever it sets.

A layout with nothing to print (no entity but its viewports, and no viewport that is on, plotted and has a view, one on a no-plot layer included, #1577) is
`SheetSkip::Empty`; one whose content is all excluded is `EmptyAfterView` (so is a layout holding only its paper view, number 1 showing
paper space 1:1 centred on itself, which is not converted at all, #1583); a file an earlier run left at either's path is removed. Any other failure of one sheet is `Failed`,
its path is left as it was, and the rest go on. An index an earlier run left is removed before the first sheet.
The whole set fails only when the destination is not a directory, the stem is missing, the drawing does not open, a selected layout does not
exist, the index cannot be written, or the run is cancelled. Each sheet publishes atomically and the set does not: `<stem>.sheets.json`
(`schema_version` 1: per layout its name, `tab_order`, `nn`, file, status `written`, `empty`, `empty-after-view` or `failed`, the reason,
the entities written and the report's `sheet` section) is written last, so its presence says the run finished. A drawing with no paper
layout writes nothing, not even the index, and never falls back to model space; one whose layouts are all empty writes the index.

`paper_sheets::<P>(source)` (#1622) lists the same paper layouts in the same tab order, each with `skip` set by the same rule (`None`, `Empty`
or `EmptyAfterView`), and converts and writes nothing. It is for a caller that makes one page per sheet and has to know how many there are
before converting the first (iasbuilt/server makes one plan per print layout at upload and converts each when it is opened). A layout listed
as a sheet can still come out `EmptyAfterView` when converted, since only the conversion knows that none of its content survives the profile.

```rust
for s in paper_sheets::<AcadrustProvider>(CadSource::Path(&dwg))? {
    if s.skip.is_none() { println!("{} {}", s.tab_order, s.layout); }
}
```
