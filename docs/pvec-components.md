# Components, style and provenance

A component is information that rides beside the geometry, never inside it. Geometry stays the PVec curve; style, fills, classes and
provenance are separate streams a viewer may drop without changing a single coordinate.

## What exists

- **Source provenance.** `pvec::provenance` keeps the handle, class name, layer and the INSERT chain of every entity in a bounded side
  store, behind an optional handle on the record. The conversion report carries the same facts per row.
- **Line style.** `pvec::style` reads colour and lineweight from the provider (BYLAYER already resolved, BYBLOCK left for the INSERT chain)
  and records a monochrome line style, refusing a degenerate dash pattern. It is recorded, not yet written as a component stream.
- **Text.** Not carried in v1: text is counted and reported, with the font outcome (embedded, missing SHX, unresolved) in
  `docs/pvec-text-policy.md`.
- **Hatch and fill.** Geometry only; the fill rule stays on the source side. See `docs/pvec-hatch-policy.md`.
- **Semantic class.** Walls and columns are classified from the DWG (`tests/pvec_classify_1233.rs`), the class stream itself is PVec's.

## What does not exist yet

The component and provenance streams of the artifact (N6.6) and the derived room and corridor regions (N6.9) are not built (#1093). So
the size split in `docs/pvec-benchmarks.md` has zero bytes for components and provenance, and the artifact carries no style: that is not a
choice about style, it is the work that is open.

When they land, the rule stays: a stream is droppable on its own, and the geometry is byte for byte the same with and without it.
