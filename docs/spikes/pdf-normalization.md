# Vector-PDF normalization and the DWG-to-PDF loss model (N1.6)

Spike for #1045, part of the integration epic #1039. It answers two questions before any adapter is
written (N5.1 to N5.3): can a vector PDF reach PVec as curves, and what has a DWG-derived PDF already
thrown away before libviprs ever sees it. This is panel finding 9: PDF equivalence holds only where the
PDF retained it.

I want to be upfront about one thing. Four of the five acceptance criteria are met and evidenced
below. The third one (a DWG plus its exported PDF, with a measured per-class geometry-count
difference) is not, because that pair does not exist anywhere yet. I say exactly why in
"Per-class geometry-count difference", and the rest of the document is written so it does not lean on
that pair.

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
| Layer identity and state | Optional content groups, when the exporter writes them, and `BDC /OC` marked content tying paths to them. Two of three files have none | partly: present in the Tulsa set (150 optional content groups, 351 `/OC` marked-content spans), absent in both print-driver files (0). Layer on/off state is in the same structure but I did not measure it | R:ocg, R:bdc.oc |
| Block structure and insert transforms | Nothing. There is no INSERT: every insert is already expanded into paths | no: no file calls a Form XObject for vector content, and a `cm` before each path is a driver artifact, not an insert transform. A flattened block and a drawn-in-place shape look the same | R:forms.unique |
| Entity handles | Nothing | no: no content-stream operator carries an identifier, and marked content in these files is `/OC` only | R:bdc.oc |
| Model/paper distinction | One page per plotted layout. Viewports may survive as `/VP` entries | partly: three of the four Tulsa pages carry `/VP` and the rest carry none, so a viewport can be seen when written, but nothing says whether a page came from model space or a layout | R:vp_pages |
| True arc and ellipse parameters | Cubic Beziers, whatever the source entity was | no: a circle arrives as four or more cubics and the centre, radius and trim are gone. A consumer could fit and guess, which is a guess and not a recovery. A quarter circle as one cubic misses the true circle by about 0.027 percent of the radius (a standard result, not measured here), so even a perfect fit would not be exact against the DWG | S2, R:segs.cubic |
| Spline degree and knots | Cubic Beziers, or lines | no: a degree, a knot vector or a weight set cannot be told from a run of cubics, and a rational spline has been approximated before it reached the stream | S2, R:segs.cubic |
| Drawing units | Page size in points and an optional `/UserUnit` or viewport measure | partly: only when the exporter writes `/UserUnit` or a measure dictionary. None of the three files declares `/UserUnit`, so for them it is no. The measure dictionary I did not look for | R:user_unit_pages, R:vp_pages |

Reading the table: a consumer of a PDF-derived drawing can recover almost nothing about what the
source was, only what it looked like. The two partly rows (layers, units) depend entirely on how the
exporter was configured, and the file in hand tells you whether it was. Nothing in the first table of
this document changes that: cubic and line semantics are exact, and what they are exact about is the
PDF's page, not the drawing.

## Per-class geometry-count difference

Not met. This criterion needs a DWG and the PDF exported from it, measured per class on both sides,
and no such pair exists yet. It comes from `libviprs/pvec-gold-fixtures#1` (Epic O), which is open
and has a README and nothing else, and no DWG-writing SDK is qualified (O1.1 and O1.2 are open).

I searched the workspace for a real pair before concluding that:

- DWGs: the acadsharp samples (`review-acadsharp-upstream/samples`, including `svg/export_sample.dwg`),
  the `wt-spdr-xdata` fixtures, and the `acadrust` test files. None has a PDF export next to it.
  `pdf-definition.pdf` among the samples is a PDF underlay sample (an image page, no paths, which the
  harness confirms), not an export of any DWG.
- PDFs: `libviprs/pdf-candidates` (arch, iso, us) and the PDFs under `iasbuilt/server/uploads`. None
  has a DWG, and none records DWG provenance in its producer fields.

I did not build a pair. A PDF I wrote from a DWG's entity list would measure my exporter, not an
AutoCAD-class one, and the per-class difference is only worth having from a real export. What I can
give instead is the PDF side of the count on three real plots, in the table above (lines, cubics,
rects, compound paths), so the day a pair exists only the DWG side and the difference are missing.
The follow-up is #1240, and the test that checks it is
`ac3_a_dwg_and_its_exported_pdf_were_measured_per_class`, ignored with that blocker in its message.

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

- The DWG plus exported PDF pair, and so the measured per-class geometry-count difference. The reason
  and the search are above, and #1240 tracks it. Because of this I do not claim that #1045 is done.
- That any of the three real PDFs came from a DWG. Their producer fields say print driver and
  Acrobat, nothing about AutoCAD, so the loss table is grounded in real PDFs but not in a real DWG
  export. In particular the arc row's "a circle arrives as cubics" is reasoned from the format, with
  `segs.cubic` showing cubics exist and not that they were circles.
- Layer state (on, off, frozen) in optional content, and `/Measure` dictionaries: I counted groups
  and viewports, not their settings.
- Form XObject recursion on a real file: none of the three has one that carries vector content.
- Operand precision. `lopdf` reads operands as `f32` (S12). A coordinate written with four decimals
  past 1,000 points has more digits than `f32` keeps, so "exact" is relative to `lopdf`. Whether that
  matters for the numeric policy is a PVec question, and a real fix is our own operand tokenizer. I
  measured the effect on a synthetic operand only, not on the fixtures.
- Rosetta: nothing ran under emulation. The x64 results are native x86_64 and the arm64 results are a
  linux/arm64 container, and the two agree.
