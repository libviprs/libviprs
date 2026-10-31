# Hatch and fill policy

A fill is area display semantics. The convenient way to ship one is to tessellate the region into triangles, and that is the thing I am not doing: mixing display triangles into mathematical geometry is panel finding 6. So every fill construct below either leaves the geometry it already has, or is counted and reported as not carried. No answer in this document produces a triangle or a point set.

The code is `src/pvec/fill_policy.rs`. `FILL_POLICY` there is the same table as this one, and `tests/pvec_1085_fill_policy.rs` reads this file back and fails if a construct or a reason code is missing from it.

## What this covers and what it does not

Two things this task depends on are not in the tree yet, and I did not fake either of them.

- The component stream (P3.7, emitted by N6.6) does not exist. So no row below emits a fill component. Where an answer would have been "geometry plus a fill component", the decision is geometry only, and the winding rule stays on the source side (`PdfPaintIntent::fill`) for N6.6 to pick up.
- The O2.1 and O2.2 fixtures (EPIC O's corpus, with `expected.math.json`) are not integrated. The counts below come from drawings and streams that are committed in this repo: `tests/fixtures/cad/g13/g13_hatch.dwg` and `g13_solid.dwg`, and content streams built in memory in the test file. When N2.1 and N2.2 land, the counts should be re-taken on their named fixtures.

## Decisions

| Construct | Decision | Reason code | Report bucket | Count and fixture |
|---|---|---|---|---|
| DWG `HATCH`, solid fill | ignored, counted | `fill-hatch-not-in-v1` | `IgnoredByPolicy` | 5 on `g13_hatch` (model space) |
| DWG `HATCH`, pattern fill | ignored, counted (same row, the provider does not yet say solid from pattern) | `fill-hatch-not-in-v1` | `IgnoredByPolicy` | 0 separately, see below |
| DWG `SOLID` | ignored, counted | `fill-solid-not-in-v1` | `IgnoredByPolicy` | 3 on `g13_solid` (model space, the two INSERTs not expanded) |
| DWG `TRACE` | ignored, counted | `fill-trace-not-in-v1` | `IgnoredByPolicy` | 0 (no fixture has one) |
| PDF `f F B b` (nonzero) | geometry only | `fill-geometry-only-nonzero` | `IgnoredByPolicy` (the fill, not the geometry) | 4 in the test stream `f F B b` |
| PDF `f* B* b*` (even-odd) | geometry only | `fill-geometry-only-even-odd` | `IgnoredByPolicy` (the fill, not the geometry) | 3 in the test stream `f* B* b*` |
| PDF `sh` shading | ignored, counted | `fill-shading-requires-extension` | `RequiresExtension` | 2 in the test stream |
| PDF `/Pattern` colour (`scn`, `SCN` with a name) | ignored, counted | `fill-pattern-requires-extension` | `RequiresExtension` | 2 in the test stream |
| PDF image XObject (`Do` of an `/Image`) | counted with its CTM, no placement | `image-pdf-xobject-not-placed` | `RequiresExtension` | 2 in the #1328 test stream |
| PDF inline image (`BI ... EI`) | counted with its CTM, no placement | `image-pdf-inline-not-placed` | `RequiresExtension` | 2 in the #1328 test stream |
| DWG `IMAGE` | reported with its frame, no placement | `image-dwg-frame-only-not-placed` | `Unsupported` | 1 in the #1328 test drawing |

### DWG `HATCH`, `SOLID`, `TRACE`

The provider maps none of the three: they arrive as `RawGeometry::Unmapped` and `FillReport::count_dwg_entities` counts them by class name. I chose "ignored, counted" for all three.

For `HATCH` the candidate was boundary curves as geometry. A boundary is a list of edges (lines, arcs, ellipses, splines) and a kept boundary would have to come out as the same curve type as the standalone entity, checked against `expected.math.json`. That needs the reader's hatch boundary data and an independent oracle, and neither is here, so I am not guessing at it. A hatch's boundary is also usually already drawn as ordinary linework, so leaving it out does not lose the outline in the common case. A follow-up issue tracks the boundary-only answer.

For `SOLID` and `TRACE` the candidate was exact triangle or quad geometry. A SOLID is a filled quad, and the quad's corners are not a curve in PVec's model, so carrying it as geometry means inventing an area construct. Counted and reported is the honest v1 answer.

The solid versus pattern split of `HATCH` needs the fill kind from the reader, which `RawEntity` does not carry. Until it does, both are one counted row.

### PDF fills

`f F f* B B* b b*` already produce subpath geometry (N5.2) and the winding rule (N5.3 records it). The geometry is unchanged by whether the path was stroked or filled, which a test checks by dumping the same path painted with `S`, `f`, `f*` and `B*` and comparing the bytes. A filled Bezier stays a Bezier. The fill intent is not carried into the canonical geometry, so these rows count the fills that were dropped, with the rule, so the report says how many and under which rule.

### PDF shadings and patterns

`sh` and a colour operator that selects a named pattern are counted (the extraction now keeps `shading_paints` and `pattern_selections` on `PdfVectorPage`) and reported as `RequiresExtension`. "Not in v1, counted and reported" was the expected answer here. Pattern rendering, gradient interpolation, transparency and blend modes stay out of scope.

### Images (issue #1328)

An image is not a fill, but the report that counts the fills is where "this page had pictures and none of them were carried" belongs, so the three image constructs sit in the same table. `PdfVectorPage::images` lists each painted image in content-stream order with the CTM that maps its unit square into page space (and the form path for an image inside a form). A DWG `IMAGE` arrives as `RawGeometry::Unmapped` with its frame in the reason (insertion point, U and V vectors, size in pixels, whether `SHOW_IMAGE` is set and whether its definition resolves as loaded). No placement is written for any of them yet: the PDF bake rules (`/SMask`, `/ImageMask`, `/Indexed`, ICC and Lab colour, `/Decode`, the CTM and the page flip) are the follow-up, and a DWG image has no pixels any reader here will fetch.

## Report

`FillReport` holds the counts per construct and `rows()` yields each non-zero one with its policy row, so a report line has a count, a bucket and a reason. It is a separate report from the fidelity tally for now: the buckets reuse `FidelityClass`, and no new `ReasonCode` was added, so a wiring into the tally can happen where the tally is assembled.

## Acceptance, as it stands

| Criterion | State |
|---|---|
| A row for every construct, with decision, reason and a count from a named fixture | done, counts on in-tree fixtures (O2.x not integrated) |
| No fill becomes triangles or a point set in canonical geometry, asserted by reading the artifact back | done for the PDF path (dump of normalized entities, entity types checked) |
| A kept hatch boundary has the same curve type as the standalone entity | not applicable: no boundary is kept in v1 (follow-up) |
| Every ignored fill appears in the report with a count and a reason | done, counts equal the fixtures' known counts |
| Fill components can be dropped without changing canonical geometry bytes | done as geometry bytes independent of paint operator; no component exists yet to drop (N6.6) |
