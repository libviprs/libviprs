# ADR: What a vector PDF may claim, and the DWG-to-PDF loss model

Status: Frozen by N1.8 (#1047). Decides N1.6 (#1045) and its follow-up (#1240), part of the
integration epic #1039.
Evidence: [docs/spikes/pdf-normalization.md](../spikes/pdf-normalization.md), with
`pdf-normalization/results-x64.tsv`, `results-arm64.tsv` and `results-pair.tsv`.

## Context

Panel finding 9: PDF equivalence holds only where the PDF retained it. `src/pdf.rs` extracts no
vector paths today, so everything here is new capability (N5.1 to N5.3). The numeric policy is P1.2's,
and libviprs defines no epsilon of its own. The conformance classes (structural exactness, geometric
exactness, display fidelity) are EPIC P's (libviprs/pvec#1), cited here and not redefined.

## Decision

- A vector PDF reaches PVec as lines and cubic Beziers, through the operator table of the spike
  (`m l c v y re h`, with `cm` and `q`/`Q` folded through `Transform2::then`). Compound paths carry
  their winding and fill rule and do not interpret them.
- A DWG-derived PDF is not assumed to preserve CAD semantics. A consumer holding only the PDF can
  tell almost nothing about what the source was, only what it looked like.
- What a PDF-derived result may claim: **geometric exactness against the PDF itself** (the locus of
  the lines and cubics in the content stream, as read), and **display fidelity against the originating
  DWG**, with an error budget measured by N5.7's differential comparison.
- A PDF-derived result is never labelled structurally exact against the DWG, because degree, knots,
  weights and trims of the source arcs, ellipses and splines are not in the PDF in those terms. A
  display-perfect result is never mathematically exact.

The loss model, by what the export threw away (rows hold for all three real PDFs unless noted):

| lost | what survives | detectable from the PDF alone |
|---|---|---|
| layer identity and state | optional content groups when the exporter wrote them | partly |
| block structure and insert transforms | nothing, inserts are already expanded | no |
| entity handles | nothing | no |
| model versus paper space | one page per layout, viewports sometimes | partly |
| true arc and ellipse parameters | cubics or short lines, by exporter | no |
| spline degree and knots | cubics or lines | no |
| drawing units | page size, sometimes `/UserUnit` or a measure | partly |

## Numbers

Three real PDFs plus one DWG and PDF pair, native x86_64 and native arm64 (they agree):

- Optional content: 150 groups and 351 `/OC` spans in the Tulsa set, none in the two print-driver
  files. Layer state itself was not measured.
- Every vector `cm` is an identity or a translation, apart from one reflection (the page y flip) and
  76 non-uniform scales that are the 76 image placements. No rotation, uniform scale or shear.
- 1,035 of 4,456 clips in the fire station set are not rectangles.
- The open-exporter pair (acadrust DWG, DXF through acadrust's writer, LibreCAD 2.2.0 plot), 31 DWG
  entities on 3 layers, gave 248 paths, 10,857 line segments, 16 cubics, 2 even-odd fills, 0 optional
  content groups. Per class: a circle became exactly 4 cubics, 5 arcs became 7,291 short lines with no
  cubics, 3 ellipses became 6,018 lines, 3 splines became 117 one-segment paths (39 each), the 4
  polylines became 16 one-segment paths and the two closed ones lost their closed flag.
- `lopdf` reads operands as `f32`, so "exact" is relative to that read.

## Consequences

- N5.1 to N5.3 implement the operator table and do not lift the harness mapper. N5.7 is asked for
  display fidelity only, never structural exactness.
- Layers and units from a PDF are used only when the file says it kept them, and the report says
  which.
- The wording of the two permitted claims is EPIC P's class names, so a report never invents a
  fourth.

## Open items

- The pair is an open-exporter pair and not AutoCAD-class: its loss is acadrust's DXF writer plus
  LibreCAD's import and plot, and the DXF step was not measured on its own. An AutoCAD or other
  plotter may keep arcs as cubics or layers as optional content. An AutoCAD-class pair belongs to
  Epic O (libviprs/pvec-gold-fixtures#1) and is unmeasured.
- Whether any of the three real PDFs came from a DWG is not established (producer fields say print
  driver and Acrobat). Layer on/off state and `/Measure` dictionaries were counted, not read.
  Form XObject recursion on a real file was not measured, none of them carries vector content.
- The panel has not confirmed the wording of the two permitted claims, and N5.1 should wait for that
  before it writes them into a report.
- Operand precision, the clip policy for non-rectangular clips (N5.2, N5.3) and fill-rule meaning
  (N5.6) are handed on and undecided. The x86_64 pair run was by a reviewer, not me, and no x86_64
  pair results file is committed.

## Images (2026-10-31, issue #1332)

Image XObjects and inline images are no longer only counted. `pvec::pdf_image` bakes them into PVec
rasters under ADR 0018 (libviprs/pvec): colour spaces into sRGB, `/Decode`, `/ImageMask`, `/SMask`
into alpha, clips into alpha plus `visible_box`, and the unit square through the CTM into a
`Transform2` placement. Anything it does not take is refused by name and counted, so the images on a
page always equal the written plus the refused. The rules and their limits are in
`docs/pvec-raster-producer.md`, and the producer rules page of libviprs/libviprs#1333 renders them.

## Image codecs (2026-11-01, issue #1366)

This answers open question 5 of epic #1329: pdfium is not required for image streams, and I wired no pdfium decode route. pdfium has no raw-stream decode call (its image API works on page objects after it has loaded the document and applied its own `/Decode`, masks and colour conversion, so it cannot sit behind a bytes-in seam and would replace the bake rules), it costs 11 to 73 MB of native artefact, and it has no wasm32 story. The decoders are small Rust crates, chosen by feature at compile time with one module per codec: `hayro-ccitt` in every build, `hayro-jpeg2000` behind `jp2k`, and `hayro-jbig2` behind the non-default `jbig2` feature (the crate says it is under development, so it stays off by default; it should not be enabled on untrusted PDFs yet, because `hayro-jbig2` can still allocate several GB (6.4 GB in #1550) or never return (#1700) (measured on 0.3.1, and 0.4.0 fixes none), upstream `LaurenzV/hayro` #1700, #1701, #1598 and #1550 being open). `jbig2dec` (AGPL) and `openjpeg-sys` (C) are out. A pdfium whole-page render as a separate fallback producer (option E) is undecided, and gets its own issue if wanted. The per-codec story and the corpus counts are in `docs/pvec-raster-producer.md`.

## Optional content (2026-11-21, issue #1637)

The vector path used to draw every `/OC` span whatever its state. It now reads `/OCProperties` and drops what the default configuration `/D` hides when viewing: the paint, images, shadings and text of a hidden `BDC /OC` span, and a form or image whose own `/OC` is hidden. A clip set inside a hidden span still applies (it is kept as an entity that paints nothing), so the clips of what follows do not move. A PDF with no `/OCProperties` is read exactly as before.

`pvec::pdf_layers` adds `pdf_manifest(bytes)` (pages, groups with `on`, `viewHidden`, `printHidden`, the `/Order` path, radio groups, direct span counts and `resolved { view, print }`, the OCMDs content refers to, and the `/Configs` names, listed and never applied) and `apply_pdf_layers(bytes, choices)`. The rewrite is an incremental update appended to the original bytes: `/D` gets `/BaseState /ON` with explicit `/ON` and `/OFF`, and `/AS` is removed. Overrides win over the mode, and forcing a group in a radio group visible turns the others of that radio group off unless they are overridden too.

Two decisions the issue did not spell out. A group whose `/Intent` has neither `View` nor `All` is always shown, because viewers leave it alone. And the rewrite also drops `ViewState` and `PrintState` from the usage of the groups that carry them: pdfium reads those keys without asking `/AS` (from its source, not measured here on a pdfium build), so stripping `/AS` alone would let it hide a group the user forced on, and the raster tiles and the PVec would disagree.

