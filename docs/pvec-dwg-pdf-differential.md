# DWG to PDF differential (N5.7)

A PDF exported from a DWG is not assumed to keep CAD semantics. This suite measures what the export lost,
on purpose. It is not an equivalence test: the DWG is the reference, the PDF is a derivative, and the result
is a recorded loss profile (`tests/pvec_dwg_pdf_differential.rs`, issue #1405).

## The exporter

The first real exporter is GNU LibreDWG's `dwg2SVG` followed by `rsvg-convert` for the SVG to PDF step, and
`qpdf --rotate` when a pair is meant to carry `/Rotate`. All of it runs in a throw-away container on MARS
(`tools/dwg-pdf-export/export.sh`, image recipe in the `Dockerfile` beside it, LibreDWG 0.14 pinned by
tarball sha256 and the base image by digest). LibreDWG is GPL, so it exists only in that image and meets
libviprs through files. Each pair's `exporter.txt` names the exporter, the versions and the options, and
the platform (native x86_64).

The exporter is a concrete script, not a trait. There is one backend; a trait that one script implements
would be a pass-through wrapper, so the seam is extracted when a second exporter exists. Swapping the
exporter means changing `export.sh` and re-recording the loss files: nothing in the Rust side names it.
The acadrust default for the DWG side is unchanged (`docs/pvec-providers.md`).

## What is committed, and what is compared

`tests/fixtures/pvec/pdf_loss/<id>/` holds `source.dwg` (from `libviprs/pvec-gold-fixtures`, `r0-open`, MIT),
`exported.pdf`, `exporter.txt` and `expected_loss.toml`. The gate needs no Docker. The pairs are
`REAL_WORLD_DIFFERENTIAL`, never `KNOWN_TRUTH_EXACT`; a test fails if any file labels them otherwise.

| axis | DWG side | PDF side |
|---|---|---|
| primitive counts | straight and arc segments of each basic entity | line and cubic segments |
| buckets | N3.5 class of each basic entity | N3.5 class of each drawn segment |
| extent aspect | exact bounds (N6.4) of the entities, width over height | the same over the painted paths, page origin removed and `/Rotate` composed by the extractor |
| outside the page | 0 | paths whose exact bounds leave the page box |

Aspect is compared to three decimals. A metric where the sides differ must be in `expected_loss.toml` with
the DWG value, the PDF value and a written reason. An unlisted difference, a wrong recorded value, an entry
for a metric that no longer differs and an entry with no reason each fail the run, and so does a missing
file. The rotated pair (`polyline-001-rot90`) must show the PDF aspect inverted while the DWG side is
unchanged.

The pairs: `line-001`, `circle-001`, `bulge-arc-001`, `ellipse-001`, `polyline-001` and `polyline-001-rot90`
(the same polyline with `/Rotate 90`).

## What the first run showed

* The exporter writes a circle or an ellipse as four cubic Beziers.
* It drops a polyline's bulge: `bulge-arc-001` leaves as a straight segment.
* It rounds coordinates to three decimals.
* Its page box does not enclose the drawing in all six pairs, so the PDF page clips part of an entity.
* A polyline is no longer one entity: its segments are separate.

## Not here

The rendered-image axis (`render_page_pdfium` against a render of the DWG-derived PVec) needs pdfium and is
#1406. The pairs are flat drawings; block-bearing pairs wait for a transform-aware bounds comparison.

## Regenerating a pair

```text
tools/dwg-pdf-export/export.sh OUT_DIR ID=PATH_TO_DWG[:ROTATE] ...
```

then copy `exported.pdf` and `exporter.txt` next to the `source.dwg`, run
`PDF_LOSS_PRINT=1 cargo test --features "dwg-acadrust pvec-normalized" --test pvec_dwg_pdf_differential -- --nocapture`
and update the loss file with a reason for each difference. Scratch on MARS is renamed to `<name>.superseded`.
