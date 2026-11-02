# What a vector PDF may claim

A PDF reaches PVec as lines and cubic Beziers, through the content stream's path operators (`m l c v y re h`, with `cm` and
`q`/`Q` folded through the transform). Compound paths carry their winding and fill rule and do not interpret them. The decision
and its evidence are `docs/adr/pvec-pdf-policy.md`.

## What it may claim

- **Geometric exactness against the PDF itself.** The locus of the lines and cubics in the content stream, as read.
- **Display fidelity against the DWG it came from.** With an error budget measured by the DWG to PDF differential.
- Never structural exactness against the DWG: degree, knots, weights and trims of the source arcs, ellipses and splines are not in the
  PDF in those terms. And a display-perfect result is never called mathematically exact.

## What an export throws away

An arc becomes cubics, an ellipse becomes cubics, a spline becomes cubics, blocks are flattened and layers are gone. The measured
loss, per pair, is `docs/pvec-dwg-pdf-differential.md`: six real exporter pairs, each held to exactly its recorded loss (primitive
counts, the six classes, the extent aspect with `/Rotate` composed, the entities outside the page).

## Clips and limits

Clip paths, nesting depth and compound paths are handled by the clip policy (`pvec::pdf_clip`), with exact bounds from PVec's closed forms
and a nesting limit. A soft mask and a glyph-outline clip are recorded, not evaluated. The vector reader has its own limits
(`PdfVectorLimits`: form depth, operations, graphics stack, entities, content bytes, segments per path, images), fuzzed by
`fuzz_pvec_pdf_vector`, and the clip policy by `fuzz_pvec_clip`. The cost of the clip stage is in `docs/pvec-benchmarks.md`.
