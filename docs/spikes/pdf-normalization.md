# Vector-PDF normalization and the DWG-to-PDF loss model (N1.6)

Spike for #1045, part of the integration epic #1039. It answers two questions before any adapter is
written (N5.1 to N5.3): can a vector PDF reach PVec as curves, and what has a DWG-derived PDF already
thrown away before libviprs ever sees it. This is panel finding 9: PDF equivalence holds only where the
PDF retained it.

All five acceptance criteria are now evidenced, with one qualification I want to be upfront about. The
DWG plus exported PDF pair for the third one (#1240) is an open-exporter pair, not AutoCAD-class: I
wrote the DWG with acadrust's `DwgWriter` and plotted it to PDF with LibreCAD, headless. The loss it
measures is LibreCAD's, and I do not claim it is what AutoCAD, or any other exporter, would lose. It is
in "Per-class geometry-count difference".

## What is there today

`src/pdf.rs` at `adcf1ff2` performs no vector path extraction, so everything below is new capability
and not a refactor. The entry points that exist are:

- `extract_page_image`, `extract_page_image_dpi`, `extract_page_image_with_background`,
  `extract_page_image_with_background_typed` and `extract_page_image_with_password`: walk a page's
  resources through `lopdf` and pull an embedded image out, or (with the `pdfium` feature) render.
- `render_page_pdfium` and `render_page_pdfium_budgeted`: hand the page to pdfium and get pixels back.
- `pdf_info` and `pdf_info_with_password`: page count, sizes and encryption state, through
  `lopdf::Document::load`.

Every one of them that produces pixels returns a `Raster`. Nothing reads a content stream for path
construction operators. The N1.1 inventory (`docs/pvec-decode-inventory.md`) reached the same
conclusion by search, and this spike does not touch any of it: the pdfium raster path is out of scope.

## What I measured, and against what

I wrote a small harness (`pdf-normalization/harness`) that does only two things. It reads a PDF with
`lopdf` 0.36, the version libviprs already depends on, decodes each page's content stream (and any
Form XObject it calls), and maps path operators into segments: a line keeps its two end points, a
cubic keeps its four control points. Then every segment goes through PVec's own constructors,
`Polyline::new` for a line and `Bezier::new` for a cubic, and every comparison goes through
`pvec_numeric::Policy::V1`. The harness has no tolerance, flattener or evaluator of its own, and a
test fails if it grows one. The `cm` stack is folded with `Transform2::then`, PVec's own composition,
and classified into the seven classes of the N1.5 matrix (`docs/spikes/transform-matrix.md`) by
`Policy::V1` comparisons. That classification is only a tally for counting. The decision about what a
transform does to a curve stays PVec's, exactly as N1.5 recommends for N3.4.

Two kinds of evidence:

- Synthetic content streams, ids `S1` to `S12`, each a few operators with a known answer (a `v`
  operator, a rotated clip, a singular `cm`). They check the mapper itself.
- Three real vector PDFs, public-domain USACE sheets from `libviprs/pdf-candidates` (licence evidence
  in its `arch/CANDIDATES.md`). Their sha256 is the one in that file, and the run records it again:

| fixture | sha256 | producer, creator | pages |
|---|---|---|---|
| `usace_hnc_yc_11-18_large_2019.pdf` | 414802e63b491e2661550cbbede13670d8115f3ecf14e0577f4c81d8e3a5ce61 | Acrobat Distiller 11.0, PScript5.dll 5.2.2 | 4 |
| `usace_hnc_std_drawings_fire_station_set1_2021.pdf` | 3770864d9ff8083d741e1c82f57354ddc40a7a08e659959b0368f945d583ddca | Acrobat Distiller 21.0, PScript5.dll 5.2.2 | 6 |
| `usace_tulsa_bt_bnhq_sample_set.pdf` | 5d83585ccc744b6411b6dba50cdcbf29a0529849544d61d5e094bbe608aa56bb | Adobe PDF Library 9.0, Adobe Acrobat Pro 10.1.8 | 4 |

I do not know that any of these three came from a DWG. They are CAD-style plots (the candidates file
calls them vector CAD plots), and the producer fields name a print-driver route, not AutoCAD. They are
the closest real thing I have to "a PDF a CAD system plotted", which is why they carry the loss table,
but they are not a DWG-derived pair and I do not claim they are.

Results are in `pdf-normalization/results-x64.tsv`, run on native x86_64 (MARS, rustc 1.99.1) by
`pdf-normalization/run.sh`, and `results-arm64.tsv`, the same harness in a linux/arm64 container on
this Mac. The two files are identical apart from their header and trailer lines, which a test checks:
nothing here is architecture dependent.

## Operator mapping

"exact" means the PDF operator and the PVec construct describe the same thing with the same
parameters. "equivalent" means the same locus or meaning, with a parameter or a flag that changes
form. One caveat covers every row: `lopdf` hands operands over as `f32` (S12 shows `0.1` arriving as
0.10000000149011612), so "exact" means exact relative to what `lopdf` delivers. See the open items.

| Operator | Verdict | Maps to | Evidence |
|---|---|---|---|
| `m` | `exact` | starts a subpath at a point; no geometry of its own | S1 |
| `l` | `exact` | a line segment between two points, a run of them is a `polyline`, with no zero-length edge allowed by NP-06 (PVec refuses one, see the findings) | S1, S10 |
| `c` | `exact` | a cubic Bezier, kept as its four control points: PVec `Bezier` of degree 3, or the same curve as a clamped `b_spline_curve_with_knots`; never flattened | S2 |
| `v` | `exact` | a cubic whose first control point is the current point, stored as a cubic and never degree-reduced; never flattened | S3 |
| `y` | `exact` | a cubic whose second control point is the end point, stored as a cubic; never flattened | S4 |
| `re` | `exact` | the spec defines it as `m l l l h`: a closed four-vertex polyline. The fact that it was a rectangle is provenance only | S5 |
| `h` | `equivalent` | a closed flag on the subpath plus, when the current point is not the start, one straight closing edge that is real geometry and goes through PVec like any other line | S6 |
| `cm` | `exact` | composed into the CTM with `Transform2::then`, classified by the N1.5 matrix. A singular or out-of-range matrix is the N1.5 refusal (`Unsupported`, `TRANSFORM_SINGULAR`), counted and never applied | S8, S11 |
| `q` | `exact` | pushes the CTM; no geometry. An unmatched `Q` is counted and ignored | S8 |
| `Q` | `exact` | pops the CTM | S8 |
| `W` | `equivalent` | a clip of the next path with the non-zero rule. Exact case: one axis-aligned rectangle under an axis-aligned CTM, carried as bounds. Bounded-only case: anything else, kept as a bounding box and never applied as a boolean intersection in v1 | S9 |
| `W*` | `equivalent` | the even-odd twin of `W`, same split. A rectangle clips identically under both rules, so the rule only matters in the bounded-only case | S9 |

The painting operators (`S s f F f* B B* b b* n`) are not on the issue's list and carry no geometry,
but the mapper needs them to know where a path ends. They are counted in the results, and the fill
rule (`f*`, `B*`, `b*` against the rest) is kept as provenance because it decides winding for compound
paths (S7).

### Compound paths and winding

A path with more than one subpath is a compound path. The mapper keeps the subpaths in order and
records the fill rule. Whether two subpaths wind in opposite directions (a hole) is decided from the
signed area of the subpath's end points, as a tally only, and S7 pins it down. On real data this
matters: of 3,634 compound paths in the Tulsa set, 3,565 mix winding, which is what a glyph or a
ring looks like once text is outlined. PVec has no fill semantics, so winding and the fill rule are
carried as provenance for N5.6 (hatch policy) and are not interpreted here.

### The CTM and N1.5

All three real files use `cm` the same narrow way, and it is a finding in itself. Every vector `cm` is
an identity or a translation (a translation ahead of nearly every path, which looks like the plot route's habit, though I did not establish why), apart from one
reflection in the fire station set (the page y flip) and 76 non-uniform scales in the Tulsa set, which
are the 76 image placements. No rotation, uniform scale or shear shows up in any of them. So the
N1.5 matrix is fully exercised by the synthetic checks and only its translation row by real data.
Composing a stack of these is exact in the N1.5 sense, and `Transform2::then` never refused one
(`ctm.refused` is zero everywhere).

## Measured on real PDFs

`lines` counts every explicit `l` plus the three non-closing edges of each `re`. The closing edges of
`h` and of `re` are counted separately (the `segs.close` rows in the results), so
accepted + refused equals lines + closing edges + cubics for each file.

| Fixture | paths | subpaths | lines | cubics | rects | accepted by PVec | refused by PVec |
|---|---|---|---|---|---|---|---|
| `usace_hnc_yc_11-18_large_2019.pdf` | 21,560 | 21,580 | 54,512 | 188 | 156 | 51,963 | 2,913 |
| `usace_hnc_std_drawings_fire_station_set1_2021.pdf` | 19,542 | 23,121 | 59,164 | 807 | 4,705 | 67,752 | 503 |
| `usace_tulsa_bt_bnhq_sample_set.pdf` | 220,249 | 224,626 | 323,271 | 110,021 | 445 | 411,445 | 22,292 |

What the numbers say:

- Cubics are preserved. For every file the count of `c`, `v` and `y` operators equals the count of
  cubic segments the mapper produced, and the harness has no code path that subdivides one. The Tulsa
  set is the heavy case: 110,021 cubics, which is outlined text and curved linework.
- Every refusal is PVec's, and there are only two kinds. `NP-06` (a zero-length line) accounts for
  all of the line refusals (2,913, 503 and 22,182), and `NP-08` (a degenerate control hull) for 110
  Tulsa cubics. A zero-length `m l` with a round cap is how a plotter draws a dot, so dropping these
  silently would lose visible marks. What the adapter does with them is an N3.5 classification and
  an N5.1 decision, and I flag it under the open items instead of choosing here.
- Clipping splits cleanly. Fire station: 3,421 exact and 1,035 bounded-only. The yc_11-18 set: 27 and
  27. Tulsa: 1 and 1. The count of exact plus bounded-only always equals the count of `W` plus `W*`.
- The nesting is shallow: the deepest `q` stack is 2, nothing is unbalanced, and none of the three
  files calls a Form XObject for vector content, so no recursion into forms was exercised on real
  data (the harness does recurse, to a depth of 8, and is unchecked on a real form).

## The loss table

Rows are what the issue lists. "Detectable from the PDF alone" says whether a consumer holding only
the PDF can tell the loss happened. It does not say the lost thing can be recovered. Evidence ids are
the synthetic checks (`S`) and the per-file measurements (`R:` followed by a results key, measured
on all three real PDFs). Where a row rests on reasoning about the format and not on a measurement, I
say so in the cell.

| Lost in the export | What the PDF keeps | Detectable from the PDF alone | Evidence |
|---|---|---|---|
| Layer identity and state | Optional content groups, when the exporter writes them, and `BDC /OC` marked content tying paths to them. Two of three files have none | partly: present in the Tulsa set (150 optional content groups, 351 `/OC` marked-content spans), absent in both print-driver files (0). Layer on/off state is in the same structure but I did not measure it. Re-checked against the pair: the DWG has three layers and LibreCAD's PDF has 0 optional content groups and 0 `BDC`, so with that exporter the layers are gone entirely, which is the `partly` verdict landing on its no side | R:ocg, R:bdc.oc, P:pair.pdf:ocg, P:pair.pdf:ops.BDC |
| Block structure and insert transforms | Nothing. There is no INSERT: every insert is already expanded into paths | no: no file calls a Form XObject for vector content, and a `cm` before each path is a driver artifact, not an insert transform. A flattened block and a drawn-in-place shape look the same | R:forms.unique |
| Entity handles | Nothing | no: no content-stream operator carries an identifier, and marked content in these files is `/OC` only | R:bdc.oc |
| Model/paper distinction | One page per plotted layout. Viewports may survive as `/VP` entries | partly: three of the four Tulsa pages carry `/VP` and the rest carry none, so a viewport can be seen when written, but nothing says whether a page came from model space or a layout | R:vp_pages |
| True arc and ellipse parameters | Cubic Beziers or dense lines, depending on the exporter and the entity. Re-checked against the pair, LibreCAD writes each circle as exactly four cubics and writes arcs and ellipses as long runs of short lines (no cubics) | no: a circle arrives as four or more cubics and the centre, radius and trim are gone. A consumer could fit and guess, which is a guess and not a recovery. A quarter circle as one cubic misses the true circle by about 0.027 percent of the radius (a standard result, not measured here), so even a perfect fit would not be exact against the DWG. The original wording here (cubics whatever the entity was) was reasoned from the format and the pair corrects it: for this exporter an arc and an ellipse are not cubics at all | S2, R:segs.cubic, P:class_circle.pdf:segs.cubic, P:class_arc.pdf:segs.line, P:class_ellipse.pdf:segs.line |
| Spline degree and knots | Cubic Beziers, or lines | no: a degree, a knot vector or a weight set cannot be told from a run of cubics, and a rational spline has been approximated before it reached the stream. Re-checked against the pair: LibreCAD wrote the three splines as 117 single-segment line paths and no cubics, so the "or lines" half of this row is the one that happened | S2, R:segs.cubic, P:class_spline.pdf:segs.line |
| Drawing units | Page size in points and an optional `/UserUnit` or viewport measure | partly: only when the exporter writes `/UserUnit` or a measure dictionary. None of the three files declares `/UserUnit`, so for them it is no. The measure dictionary I did not look for | R:user_unit_pages, R:vp_pages |

Reading the table: a consumer of a PDF-derived drawing can recover almost nothing about what the
source was, only what it looked like. The two partly rows (layers, units) depend entirely on how the
exporter was configured, and the file in hand tells you whether it was. Nothing in the first table of
this document changes that: cubic and line semantics are exact, and what they are exact about is the
PDF's page, not the drawing.

## Per-class geometry-count difference

Met, for an open-exporter pair, not AutoCAD-class. The loss below is LibreCAD's, and I make no claim
about any other exporter. An AutoCAD-class pair is still wanted from Epic O
(`libviprs/pvec-gold-fixtures#1`), and I searched the workspace for a real one first: the acadsharp
samples, the `wt-spdr-xdata` fixtures and the `acadrust` tests hold DWGs with no PDF beside them, and
`pdf-candidates` and the iasbuilt uploads hold PDFs with no DWG.

Provenance, all recorded in `pdf-normalization/results-pair.tsv` and regenerated by `pair.sh`:

- DWG: `pair.dwg`, written by acadrust (`libviprs/acadrust` at `d35691f`) with `DwgWriter`, version
  AC1032, three layers (`WALLS`, `ARCS`, `ANNO`) and 32 entities. I read it back with `DwgReader` and
  counted the DWG side from that read-back, so the count is of the file and not of the generator.
- DXF: written by acadrust from the read-back DWG, so the PDF is an export of what the DWG holds.
- PDF: `pair.pdf`, plotted by LibreCAD 2.2.0-1 (Debian bookworm package, arm64, Qt 5.15.8, offscreen
  platform, no display) with `librecad dxf2pdf -a`. Its producer field says Qt 5.15.8, not AutoCAD.
- sha256 of the three files is in the results file. Everything ran in linux/arm64 containers.

LibreCAD's PDF has no entity classes in it, so I did not attribute path counts to classes by guessing.
Instead the generator also writes one DWG and DXF per class and each is plotted on its own, so every
PDF row below is measured for that class alone. Every LibreCAD PDF carries one rectangular clip (a `W*`
path of four line segments), which I subtract from the line and path counts below.

| class | DWG entities | PDF paths | PDF line segments | PDF cubics | PDF fills | difference |
|---|---|---|---|---|---|---|
| line | 6 | 6 | 6 | 0 | 0 | none in count: one entity, one path, one segment |
| polyline | 4 (18 vertices) | 16 | 16 | 0 | 0 | the 4 polylines became 16 one-segment paths, so polyline identity is gone, and the two closed ones lost their closed flag (no `h`, the closing edge is an explicit line) |
| arc | 5 | 5 | 7,291 | 0 | 0 | 5 arcs became about 1,458 short lines each, no cubics |
| circle | 4 | 4 | 0 | 16 | 0 | exactly four cubics per circle, closed |
| ellipse | 3 | 3 | 6,018 | 0 | 0 | lines, no cubics, and the partial ellipse is just a shorter run |
| spline | 3 | 117 | 117 | 0 | 0 | 3 splines became 117 one-segment paths (39 each), no cubics |
| text (outlined to paths) | 4 | 95 | 3,311 | 0 | 0 | LibreCAD strokes its own line font as paths, it does not outline glyphs, so 4 text entities are 95 paths |
| hatch | 2 | 2 | 8 | 0 | 2 (even-odd) | one filled path per hatch, boundary kept as four lines |
| other | 0 | 0 | 0 | 0 | 0 | nothing present |

The whole drawing in one PDF has 248 paths, 10,857 line segments, 16 cubics, 2 even-odd fills, 0
optional content groups, 0 `BDC` and no `/UserUnit`. The per-class segment counts do not add up to
10,857 (they add up to 16,767) because each class was plotted with fit-to-page on its own, and
LibreCAD's line density for curves follows the plot scale. So the line counts for arcs, ellipses and
text are true for those plots and are not apportionments of the whole-drawing plot. I have not worked
out why the arc density is as high as it is, and I would not read anything into the exact numbers beyond
"a long run of lines". Measured on arm64 only. The harness counts did not need a second architecture
for this, but I did not run an x86_64 pass.

What this says about the loss table: the layer, handle, block and units rows hold as written, the arc,
ellipse and spline rows had to be corrected (above), and nothing in the pair contradicts the rest.

## Labelling

A PDF-derived result is never labelled structurally exact against the originating DWG. Structural
exactness compares degree, control points, knot values and multiplicities, weights and trim, and the
loss table shows that a DWG's arcs, ellipses and splines are not in the PDF in those terms, so the
comparison is not available to make.

The EPIC P conformance classes (libviprs/pvec#1) are structural exactness, geometric exactness and
display fidelity. What a PDF-derived result may claim:

- Geometric exactness against the PDF itself: the locus of the lines and cubics in the page's content
  stream, under `Policy::V1`, as read. That is a claim about the PDF page, and it is bounded by the
  `f32` operand caveat below.
- Display fidelity against the originating DWG, with an error budget measured by the differential
  comparison of N5.7. This is the class it may claim for the drawing, because an arc that became
  cubics has a different locus from the arc it came from, and a display-perfect result may never be
  labelled mathematically exact.

I want the panel to confirm that wording before N5.1 puts it in a report, since the class names are
EPIC P's and not libviprs's.

## Handed on

- N1.8 and N5.1: the operator table is the contract. Read operators with `lopdf`, map `m l c v y re h`
  as above, fold `cm` and `q`/`Q` through `Transform2::then` and call the N1.5 matrix for the result.
  The mapper in the harness is a throwaway and should not be lifted: it exists to measure.
- N5.2 and N5.3: clipping needs a decision on the bounded-only case. 1,035 of 4,456 clips in the fire
  station set are not rectangles, so the choice (ignore, record bounds, or intersect with a PVec
  capability that does not exist) changes what the output looks like.
- N5.6: winding and fill rule are carried, not interpreted, and the Tulsa set shows why they matter.
- N5.7: this document says what the differential comparison can be asked to prove. It can show display
  fidelity of a PDF against the DWG it came from, and it cannot show structural exactness.

## What I could not verify

- An AutoCAD-class pair. The pair that exists is the open-exporter pair above (acadrust DWG, LibreCAD
  PDF). Its loss is LibreCAD's, and a PDF from AutoCAD or another plotter may keep arcs as cubics or
  keep layers as optional content. Epic O (`libviprs/pvec-gold-fixtures#1`) is where such a pair belongs.
- That any of the three real PDFs came from a DWG. Their producer fields say print driver and
  Acrobat, nothing about AutoCAD, so those rows are grounded in real PDFs but not in a real DWG export.
  The open-exporter pair now shows circles arriving as cubics for LibreCAD, and arcs and ellipses not.
- Layer state (on, off, frozen) in optional content, and `/Measure` dictionaries: I counted groups
  and viewports, not their settings.
- Form XObject recursion on a real file: none of the three has one that carries vector content.
- Operand precision. `lopdf` reads operands as `f32` (S12). A coordinate written with four decimals
  past 1,000 points has more digits than `f32` keeps, so "exact" is relative to `lopdf`. Whether that
  matters for the numeric policy is a PVec question, and a real fix is our own operand tokenizer. I
  measured the effect on a synthetic operand only, not on the fixtures.
- Rosetta: nothing ran under emulation. The x64 results are native x86_64 and the arm64 results are a
  linux/arm64 container, and the two agree.
