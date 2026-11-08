# The source-provider API

A reader of DWG files sits behind one trait, `pvec::DwgProvider`. Nothing past it names a reader, a decoder type or a
file format, which is what makes a provider replaceable (`docs/pvec-provider-replacement.md`).

## What a provider gives

| method | answers | when it cannot |
|---|---|---|
| `probe(source)` | whether this provider would open the source, from its first bytes | `Err` only when the source cannot be read at all |
| `open(source)` | an owned provider for the drawing | `ProviderError::Unrecognised` or `Open` |
| `name()` | the name reports use | |
| `caps()` | a `ProviderCaps`: which facets it can report | |
| `layouts()` | model space first, then paper layouts in tab order | |
| `layers()` | name, handle, off, frozen, locked, plottable, in table order | `CannotReport(LayerState)` |
| `entities(scope)` | an `EntityCursor` over a layout or a block definition | `NoSuchScope` |
| `entity_facts`, `viewports`, `xref_blocks`, `text_styles`, `layout_plot` | optional facets, default `CannotReport` | |

For a print (#1557), `layout_plot(id)` gives a layout's plot settings as stored (`LayoutPlot`: paper size and name, paper
units, plot type and window, plot scale, rotation, margins, origin, the CTB name and the lineweight, plot style, border and
draw order flags), `Absent` for a pre-R2000 layout that has none. A `ViewportEntity` carries its `id` (every viewport is listed, the one numbered 1 included, so which is the paper view is the print
profile's call), its paper rectangle, its own
layer, its view (centre, height, target, direction, twist, perspective), `hide_plot`, the non-rectangular clipping flag and a
`ViewportClip`: the viewport's rectangle when there is no boundary entity, a polygon with its bulges for an LWPOLYLINE or 2D
POLYLINE boundary, and a circle, ellipse, spline or other boundary named by handle rather than replaced by the rectangle.
Nothing is applied: the print profile decides what each one does.

A facet a provider cannot read is `CannotReport`, and a fact a drawing does not have is `SourceFact::Absent`. The two are
different on purpose: the first is about the provider, the second about the drawing, and they lead to different
outcomes downstream. `SourceFact::NotReportable` is the third arm, for a value the drawing may well have and this provider
cannot read.

## What an entity looks like

A `RawEntity` is a geometry (`Line`, `Polyline` with bulges, `Arc`, `Circle`, `Ellipse`, `Spline` with its knots, weights and
periodicity kept apart, `Insert`, or `Unmapped` with a reason), a `ProvenanceStub` (handle, class name, the INSERT chain it was
reached through, layer) and a style (colour and lineweight as the drawing states them). Coordinates are source `f64`,
unconverted. A provider hands out one block definition at a time and does not expand INSERTs: the flattener does.

## Providers that exist

`AcadrustProvider` (feature `dwg-acadrust`) is the default and the only `DwgProvider`. `acadsharp-rs` is opt-in through
`cad::CadDecoder` and is not a `DwgProvider` yet. See `docs/pvec-providers.md` and `docs/pvec-version-compatibility.md`.
