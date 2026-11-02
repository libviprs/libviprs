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
| `layers()` | name, handle, off, frozen, locked, in table order | `CannotReport(LayerState)` |
| `entities(scope)` | an `EntityCursor` over a layout or a block definition | `NoSuchScope` |
| `entity_facts`, `viewports`, `xref_blocks`, `text_styles` | optional facets, default `CannotReport` | |

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
