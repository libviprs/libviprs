# ADR: What "a paper layout printed in monochrome" means

Status: Frozen by #1559 as a rule, slice 1 of the layouts epic #1557, decided in the design comment
on #1557. Its implementation lands in slices 3 to 6 (#1562 to #1565).
Evidence: the paper space section of [docs/spikes/view-context.md](../spikes/view-context.md), the
model-to-paper formula and the exact versus bounded split measured there, and the three
known-truth fixtures listed below.

## Context

The owner wants one pvec per paper layout of a DWG or DXF, each "as if we had printed that LAYOUT
sheet in monochrome". [pvec-view-profile.md](pvec-view-profile.md) draws model space only and makes
paper space a v1 non-goal (V1-1), with no-plot and viewport-frozen layers visible (V1-3). Every
caller of `ViewProfile::default_profile()` depends on those clauses, so I'm not reopening them.
This ADR adds a second named profile beside the first, and the default stays model space.

The numeric policy is P1.2's (PVec numeric policy v1), and libviprs defines no epsilon of its own
here. Nothing in this ADR is a threshold.

## Decision

`ViewProfile::print_profile(layout)` is the print of one paper layout. `default_profile()` and
V1-1 to V1-5 stay exactly as they are, and `PvecConversion::convert()` still draws model space. A
test pins `print_profile` field by field the way one pins `default_profile`, so changing a value is
a red test and a new ADR.

- **P-1 Space.** The named paper layout contributes: its own block record (title block, border,
  notes), and model space seen through each of its viewports. Only that layout; a sheet never
  falls back to model space and is never merged with another sheet.
- **P-2 Effective layer.** V1-2, unchanged.
- **P-3 Visibility.** V1-3, plus three print rules. Entities on a no-plot layer are excluded
  (`layers.no_plot = exclude`). Viewport freezes apply inside the viewport that froze the layer
  (`layers.frozen_in_viewport = apply`). A viewport that is off, invisible or `hide_plot`
  contributes nothing. A viewport on a no-plot layer still draws what it shows; only its border
  is not printed, as AutoCAD prints it (amended by #1577, see below).
- **P-4 Instances.** V1-4, once per viewport that shows them. A model entity seen through two
  viewports is two pvec entities with distinct provenance (`source_kind` 2, the source handle
  followed by the viewport handle).
- **P-5 External and missing.** V1-5, unchanged.
- **P-6 Sheet frame and unit.** Coordinates are paper-space drawing units, never rescaled. The
  manifest unit comes from `plot_paper_units` (0 inch, 1 millimetre; anything else falls back to
  `$INSUNITS` and is reported). The manifest box is the paper sheet, `paper_width` by
  `paper_height` in the sheet unit, placed from the plot origin and the margins, so every sheet of
  a set has the same frame. `plot_rotation` is reported, not baked. A `plot_type` other than layout
  and the plot window are reported, not applied. Where the plot settings are absent or zero
  (pre-R2000, or `paper_width = 0`) the box is the kept geometry's extent and the report says
  `sheet: extents`.
- **P-7 Clipping.** Nothing is drawn outside its viewport or outside the sheet
  (`viewport_clipping = exact`). Lines and polylines are split exactly against the rectangle.
  A curve wholly inside is kept and one wholly outside is dropped as `OutsideViewport`. Since
  slice 6 (#1565) one that crosses is split: circles, arcs and ellipses by parameter interval
  (`SupportedExact`) and splines by knot insertion (`SupportedEquivalent`, with the parameter
  tolerance in the row), and a viewport clipped to a straight-edged polygon is cut by the
  polygon, then the sheet. A rational spline that crosses is still dropped and counted
  `Unsupported` with a per-entity 508, and so is a viewport whose boundary has a bulge or is a
  circle, ellipse or spline. Nothing is ever tessellated and there is no pvec clip capability.
- **P-8 Monochrome.** monochrome.ctb semantics: every visible stroke is black, no screening,
  colours 7 and 255 black, object lineweight kept in the style stream. `print_lineweights = false`
  drops widths and is reported. `scale_lineweights`, `plot_plot_styles` and the named
  `plot_style_sheet` are reported only. A model lineweight seen through a viewport is a paper width
  and does not go through the viewport map. The drawing's CTB file is never opened from disk.
- **P-9 Calibration.** The layout's plot scale is the artifact's calibration record,
  `PlanScale { numerator, denominator, paper_metres_per_unit }`, and is never applied to
  coordinates. Per-viewport scales go in the report's `sheet.viewports` rows, and when exactly one
  viewport shows model space its scale also goes in the record's note. No georeference unless the
  caller supplies one.

The profile as data, beside the N1.4 rows of the view-profile ADR. `layer-no-plot` is the one row
this ADR adds; the default profile carries it as `ignore`, which is what V1-3 already says.

| field | default (V1) | print (P) | row |
|---|---|---|---|
| `space` | model | paper:LAYOUT | `paper-space-layout` |
| `viewport_clipping` | unsupported | exact | `viewport-clip-exact` |
| `layers.off` | exclude | exclude | `layer-off` |
| `layers.frozen` | exclude | exclude | `layer-frozen` |
| `layers.no_plot` | ignore | exclude | `layer-no-plot` |
| `layers.frozen_in_viewport` | ignore-and-report | apply | `layer-frozen-in-viewport` |
| `layers.insert_override` | inherit-on-layer-zero | inherit-on-layer-zero | `insert-layer-override` |
| `nested_inserts.max_depth` | 64 | 64 | `nested-insert` |
| `xrefs` | report-external | report-external | `xref` |

An entity excluded by `layer-no-plot` is counted per layer under `LAYER_HIDDEN` (proposed), the
same counter as an off or frozen layer, so the report still tells "nothing was on Defpoints" from
"the print rule dropped it".

Until the slice that implements a print rule lands, asking for it is refused, never ignored. Since
slice 3 (#1562) a paper layout is drawn under the print rules (`viewport_clipping = exact`) and stays
`PaperSpaceUnsupported` (508) under the default ones, and `layers.no_plot = exclude` is applied
from the provider's plottable flag (#1560) in either space. On model space
`layers.frozen_in_viewport = apply` and `viewport_clipping = exact` are refused with
`RuleNotEnabled`, since model space has no viewport for them to act on. Since slice 4 (#1563) model
space shows through each viewport the profile can map; the rest are refused per viewport as below,
plus a target off the origin (the view centre is in display coordinates whose origin is the target,
and no reference print measures it), and the paper view (number 1, showing paper space 1:1 centred on
itself, as AutoCAD writes it) is not a window onto the model.

[pvec-unsupported-policy.md](pvec-unsupported-policy.md) is amended to match: paper space is no
longer `Unsupported` as a whole. `VIEW_NOT_ENABLED` (508) stays, per viewport (a non-plan view
direction, perspective, `view_height <= 0`, a clip boundary with a bulge or a curve, a twist until
its sign is measured) and per entity (a rational spline crossing a clip).

Amendment to P-3 (#1577). As first written, P-3 also dropped a viewport on a no-plot layer. That is
not what a print does: AutoCAD prints the content of such a viewport and leaves out only its border,
and sheets often put their viewports on Defpoints for exactly that reason, so the old rule turned
those sheets into a bare title block. The viewport now goes through the same checks as any other
(drawn, or refused with 508 for the same reasons), its own VIEWPORT entity is excluded by
`layer-no-plot` like anything else on that layer, and its `sheet.viewports` row says
`border: no-plot-layer`. pvec draws no viewport border on any layer, so the row is the only place the
difference shows.

Rejected: amending V1-1 and V1-3 in place (every caller of the default profile would change
behaviour); one profile with a `space` switch that flips the rules (the rules are the profile); a
pvec clip-region capability (pvec is curve-only, and a reader without the capability would draw
model geometry over the title block); millimetres always (rescales authored coordinates, against
`docs/pvec-units.md`); the geometry extent as the box (sheets of a set would not share a frame);
keeping a crossing curve whole and calling it approximate (not a fidelity class, and it draws over
the title block); a pvec change for per-region scales (reopens ADR 0019, no consumer).

Dissents kept: bake the rotation, since the 90 degree cases are exact in f64 (I'd rather report it
and let the consumer rotate); write the single viewport's scale as the file's record (a reader of
`effective_scale()` would measure the title block a hundred times too big, and `recalibrate`
exists for an application that wants it); keep lineweights when `print_lineweights` is false (a
print without lineweights has none, and the report says the flag did it).

## Fixtures

Known truth that exists, vendored under `tests/fixtures/pvec-gold/` from
`libviprs/pvec-gold-fixtures` r3-open, writer-authored, none with a twist, a freeze or a non-plan
view:

- `paper-space-multi-viewport-001`: twenty rectangular viewports on `Layout1`, slice 4. Its targets
  are at the origin since pvec-gold-fixtures#102, and it draws through the print profile since #1579.
- `viewport-clipped-001`: one polygon-clipped viewport, slice 6.
- `viewport-clipped-bounded-001`: the same clip recorded as bounded only, slice 6.

Requested from `libviprs/pvec-gold-fixtures` as new `author_probes.py` parameters: a twisted
viewport, a viewport freeze, a non-plan view direction, a `hide_plot` viewport and a no-plot
layer, and a layout with plot settings (paper size and unit, plot scale, rotation, non-zero
margins and a non-zero plot origin), the last being the one that pins the sheet position of P-6.

Asked of the owner: one AutoCAD (or ODA) plotted PDF of a layout with a twisted viewport, a
viewport freeze and a non-plan view, so a sheet pvec can be compared against `PdfConversion` of
that PDF as a differential oracle. Twisted viewports stay refused until it exists.

## Consequences

- Slices 3 to 6 implement against P-1 to P-9 and do not reopen them. A change to a clause is a new
  ADR.
- `ViewportClipping` gains `Exact` and `ViewportFreezeRule` gains `Apply`. Both enums were
  `#[non_exhaustive]` already. `LayerVisibility` gains `no_plot`, which the integration contract
  pins as `ignore` for the default profile.
- No pvec format change: the file name and the report carry the sheet name, the manifest box and
  unit carry the paper size.

## Open items

The rules are settled. These are the places where the evidence behind them is thin, each with the
measurement that is missing:

- The twist sign (DXF group 51). No AutoCAD-authored fixture with a twist exists here; unmeasured,
  so a twisted viewport is 508 until the reference print above is measured.
- Where the paper sheet sits in paper-space coordinates from the plot origin, the margins and
  `plot_rotation`. I believe the printable area's lower-left is the origin shifted by origin and
  margins, but nothing has measured it; the plot-settings fixture above is that measurement.
- Whether acadrust's reader fills every `Layout.plot_*` field for R2000 to R2018, and whether
  `plot_paper_units = 2` occurs in real files. The fields exist; not measured on real drawings.
- Whether pvec's writer refuses an entity whose bounds leave the manifest box. Slice 3 does not
  depend on it: the sheet is handed to the writer as the area its box must take in, so the box is
  the union of the sheet and the kept geometry, which the clip keeps inside the sheet anyway. Not
  measured.
- That AutoCAD cuts paper-space entities at the paper edge on PLOT. Stated from experience, not
  measured.
