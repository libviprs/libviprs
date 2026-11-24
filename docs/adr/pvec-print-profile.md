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
  is not printed, as AutoCAD prints it (amended by #1577, see below). Text is the exception to
  the frozen rule: text on a frozen layer is drawn (`layers.frozen_text = ignore`), and text on a
  no-plot layer stays excluded (`layers.no_plot_text = exclude`, like its geometry) (amended by
  #1621 and #1639, see below).
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

The profile as data, beside the N1.4 rows of the view-profile ADR. `layer-no-plot` is the row
this ADR adds; the default profile carries it as `ignore`, which is what V1-3 already says. The
two text rows were added by the #1621 amendment below. `layer-frozen-text` carries the same value
in both profiles, and `layer-no-plot-text` is `ignore` on screen and `exclude` on a sheet (#1639).
`entity-invisible` (#1633) is `exclude` in both: AutoCAD never draws an entity whose own invisible
flag is set, and a hidden INSERT takes its whole block with it. `ignore` draws them, as a
diagnostic aid. An entity dropped this way is counted under `ENTITY_INVISIBLE`.

| field | default (V1) | print (P) | row |
|---|---|---|---|
| `space` | model | paper:LAYOUT | `paper-space-layout` |
| `viewport_clipping` | unsupported | exact | `viewport-clip-exact` |
| `layers.off` | exclude | exclude | `layer-off` |
| `layers.frozen` | exclude | exclude | `layer-frozen` |
| `layers.no_plot` | ignore | exclude | `layer-no-plot` |
| `layers.frozen_text` | ignore | ignore | `layer-frozen-text` |
| `layers.no_plot_text` | ignore | exclude | `layer-no-plot-text` |
| `layers.overrides` | none | none | `layer-overrides` |
| `layers.frozen_in_viewport` | ignore-and-report | apply | `layer-frozen-in-viewport` |
| `layers.insert_override` | inherit-on-layer-zero | inherit-on-layer-zero | `insert-layer-override` |
| `entities.invisible` | exclude | exclude | `entity-invisible` |
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

Plan view and no twist are judged up to float noise (#1660): a DWG stores the view direction as
camera minus target, so a plan view carries a lean near 1e-16 rad (x and y over z) and a twist of the
same size, or one just under 2 pi. A view direction whose x and y over z, or a twist whose distance
from a whole turn, is within 1e-9 rad is plan view and untwisted; anything over that is still 508, and
so is a NaN or an infinite lean (an infinite x or y next to an infinite z). Ignoring 1e-9 rad moves a
point by that times its distance from the target, which on a 1e5 unit sheet is about 1e-4 drawing
units. The reader hands the stored values over untouched; the tolerance is in `viewport_class`, the
one place the conversion and the manifest both ask.

Amendment to P-3 (#1577). As first written, P-3 also dropped a viewport on a no-plot layer. That is
not what a print does: AutoCAD prints the content of such a viewport and leaves out only its border,
and sheets often put their viewports on Defpoints for exactly that reason, so the old rule turned
those sheets into a bare title block. The viewport now goes through the same checks as any other
(drawn, or refused with 508 for the same reasons), its own VIEWPORT entity is excluded by
`layer-no-plot` like anything else on that layer, and its `sheet.viewports` row says
`border: no-plot-layer`. pvec draws no viewport border on any layer, so the row is the only place the
difference shows.

Amendment to P-3 (#1621, Rom's call, 2026-11-19). Text on a frozen layer is drawn on
the sheet, both in the layout's own paper space and in model space seen through a viewport, and
text on a no-plot layer is left out, as no-plot geometry is (the default profile draws both,
#1639). A sheet therefore shows notes its plot lacks (superseded
schemes, demolition notes) when they sit on frozen layers; the upload dialog says so. The scope is
the same as the V1-3 amendment: the layer table's frozen flag only, so a viewport freeze still drops
text inside the viewport that froze its layer (without that, every sheet's notes would show in every
viewport); TEXT, MTEXT and visible attributes only, never DIMENSION or MULTILEADER; an off layer
still hides text; an INSERT on a frozen layer still takes its attributes with it; and the layer is
the effective one after flattening. A viewport on a no-plot layer still draws what it shows, text
on its own plotting layers included.

A comparison against a plotted PDF of a sheet therefore runs from the PDF to the sheet only: every
string the PDF prints must be on the sheet near its position, extra strings on the sheet are
reported and never fail, strings on frozen layers are counted apart, and strings on no-plot layers
are asserted absent.

Rejected for this amendment: a rule inside the annotation mapper rather than a knob in the layer
filter (two places would decide visibility); including viewport freezes; splitting DIMENSION
into text and lines; a PVec text record carrying its layer and frozen flag (a format change).

Amendment to P-6 and P-9 (#1628). The frame and the calibration can now follow the caller's plot settings (`PvecConversion::plot_settings`), and nothing changes without them. P-6: with plot area Extents the paper is placed from the extents of what the sheet draws (that corner is the origin instead of the paper-space origin); the offset replaces the layout's plot origin (one quantity, as the plot dialog shows it) and shifts the printable area's corner; centre puts the extents in the middle of the printable area (the paper less its margins), ignores the offset, and applies only to plot area Extents; a paper size override replaces the layout's paper and margins and still follows `plot_rotation`; orientation turns the paper after the rotation. Fit to paper makes the frame the window of the drawing that the paper shows, so it is no longer the paper's own size in drawing units: under Extents it fits the extents into the printable area, under Layout it fits the layout's own printable area into the new paper's (a no-op without a paper override). The report's paper fields, not its box, are the paper under fit. P-9: the calibration record is still the plot scale, the layout's or the caller's `scale` (which carries its own paper unit, and the record keeps it), and is never applied to coordinates; fit does not rewrite it, and the fit ratio is a report note. Plot lineweights and paperspace last replace the layout's flags (P-8 widths, P-1 order). Plot area Window and rotation 180 stay reported, not applied. Position is still `unmeasured`, and offset and centre compound on that unmeasured placement until the plot-settings fixture exists.

Amendment to P-6 and P-9 (#1665, 2026-11-24). A sheet's linework is drawn at its plot scale. P-6 took a paper-space drawing unit to be a paper unit, and that only holds at 1:1: a real drawing's Tabloid layouts are millimetres at 1 : 0.04, so a drawing unit prints as 25 mm, and the sheet came out as linework 17 units across inside a 431.8 mm frame. Now the sheet's own leaves, its viewport windows (the map, the rectangle and the clip polygon), its extents under plot area Extents and a plot window all go through one transform at the sheet root, `n / d` times the paper unit's size in the sheet unit, composed before the turn of a plot upside-down. That covers coordinates, text heights and anything else the leaf's map scales; lineweights are paper millimetres already and stay as they are, and `scale_lineweights` is still reported, not applied. The frame, the box and the unit stay paper, as P-6 says. It applies to the layout's own scale and to the caller's `scale` alike, a zero scale draws 1:1, and the report notes `plot-scale-applied` when the scale isn't 1:1. P-9: the plot scale is still the calibration record, with the stored ratio as its provenance, and it is now also the scale the sheet is drawn at. The record's metres per unit doesn't change, because it always counted a coordinate as a unit of paper, which the coordinates now are... so a length measured through it on a sheet at 1 : 0.04 is right where it used to be 25 times short. Fit to paper keeps the #1628 reading (the frame is the part of the drawing the paper shows, so its box isn't the paper), now over linework drawn at the scale. A layout plotted at 1:1 is byte-identical.

Rejected for this amendment: keeping the linework in drawing units and framing the sheet in them, with the manifest at 0.025 m per unit (`SourceUnit` can't name 25 mm, the viewer's descriptor names four units, and every width would be off by the same factor); leaving it to each caller (a real sheet from the server would still draw tiny); an opt-in switch for one release (the old output is wrong).

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
  pins as `ignore` for the default profile, and since #1621 `frozen_text` and `no_plot_text`,
  pinned as `ignore` and, since #1639, `ignore` (the print profile sets `no_plot_text` to
  `exclude`).
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
