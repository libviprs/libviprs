# Corpus regression report

This is the conversion equivalent of `docs/pmtiles-benchmarks.md`: every fixture of the pinned
`pvec-gold-fixtures` revision (`r2-open`, vendored under `tests/fixtures/pvec-gold/`, 57 fixtures) goes through
`PvecConversion`, and three sections are recorded for each one. The deterministic columns are the
committed baseline, `tests/fixtures/pvec_corpus_baseline.json`, and the diff against it runs in the ordinary
suite (`tests/pvec_corpus_regression.rs`, `the_corpus_does_not_regress`). Nothing in it runs a converter.

## What a row says

- **Mathematical.** The conversion classes of the entities (the six of EPIC P) and the fixture's conformance, the
  worst class any written entity landed in: `SupportedExact`, `SupportedEquivalent` (the same locus under another
  construct) or `NoGeometry`. The fixture's evidence class is printed beside it, and the baseline row says what that
  class means: a `KNOWN_TRUTH_EXACT` expectation was authored independently, a `DUAL_ORACLE_EQUIVALENT` row is two
  converters agreeing and is **not source truth**, a `REAL_WORLD_DIFFERENTIAL` row is regression only, and a
  `QUARANTINED` row runs and is reported and never gates. A display-fidelity result is never called exact here.
- **View policy.** The space drawn, the leaves kept, the leaves excluded by profile field, and the xrefs left
  unresolved.
- **Size and performance.** Entities met, entities written, `.pvec` bytes, and in the report wall time and peak RSS.

## The gate

A fixture that moves from `SupportedExact` to `SupportedEquivalent` (or loses geometry), one that stops converting,
or one that has entities fall out of `SupportedExact` fails the gate. Total `.pvec` bytes growing more than 2%, or one
fixture's growing more than 5%, fails it. A fixture that is not in the baseline fails it too, so a new fixture has to
be reviewed into the baseline. Quarantined fixtures appear in the report and never gate. The unit tests in the same file
edit a copy of the baseline to prove each of those.

## Regenerating the baseline

One line, and review the diff before committing it:

```text
LIBVIPRS_BENCH_HOST=mars LIBVIPRS_CORPUS_BASELINE_OUT=tests/fixtures/pvec_corpus_baseline.json \
    cargo test --release --features pvec-convert --test pvec_corpus_regression -- --ignored --nocapture corpus_report
```

## The current report

57 fixtures: 38 `KNOWN_TRUTH_EXACT`, 18 `QUARANTINED`, 1 `DUAL_ORACLE_EQUIVALENT`, 0 `REAL_WORLD_DIFFERENTIAL`.
34 convert; the other 23 are refused, and every refusal is in the baseline with its error chain. Total `.pvec`
bytes over the corpus: 18,024. Timings and RSS below were taken on MARS (native x86_64), release build, one run each, and are
not gated: peak RSS of the whole run 9596928 bytes.

| fixture | evidence class | conformance | leaves kept | entities | .pvec bytes | wall (ms) | platform |
|---|---|---|---|---|---|---|---|
| arc-seam-001 | KNOWN_TRUTH_EXACT | SupportedEquivalent | 1 | 1 | 344 | 9.3 | x86_64-linux-mars |
| arc-tiny-sweep-001 | KNOWN_TRUTH_EXACT | SupportedEquivalent | 1 | 1 | 360 | 2.2 | x86_64-linux-mars |
| bezier-d3-001 | KNOWN_TRUTH_EXACT | SupportedExact | 1 | 1 | 888 | 2.1 | x86_64-linux-mars |
| blocks-dynamic-linear-001 | DUAL_ORACLE_EQUIVALENT | SupportedExact | 2 | 2 | 368 | 2.5 | x86_64-linux-mars |
| bspline-d3-001 | KNOWN_TRUTH_EXACT | SupportedExact | 1 | 1 | 928 | 1.7 | x86_64-linux-mars |
| bspline-d5-001 | KNOWN_TRUTH_EXACT | SupportedExact | 1 | 1 | 944 | 2.0 | x86_64-linux-mars |
| bspline-nonunit-domain-001 | KNOWN_TRUTH_EXACT | SupportedExact | 1 | 1 | 904 | 2.1 | x86_64-linux-mars |
| bulge-arc-001 | KNOWN_TRUTH_EXACT | NoGeometry | null | 1 | 0 | 0.9 | x86_64-linux-mars |
| bulge-arc-neg-001 | KNOWN_TRUTH_EXACT | NoGeometry | null | 1 | 0 | 0.7 | x86_64-linux-mars |
| circle-001 | KNOWN_TRUTH_EXACT | SupportedExact | 1 | 1 | 360 | 5.4 | x86_64-linux-mars |
| closed-by-endpoint-001 | KNOWN_TRUTH_EXACT | SupportedExact | 1 | 1 | 912 | 2.2 | x86_64-linux-mars |
| closed-periodic-001 | KNOWN_TRUTH_EXACT | SupportedEquivalent | 1 | 1 | 976 | 8.2 | x86_64-linux-mars |
| coincident-control-points-001 | QUARANTINED | NoGeometry | null | 1 | 0 | 0.7 | x86_64-linux-mars |
| dimension-001 | QUARANTINED | NoGeometry | null | 1 | 0 | 0.8 | x86_64-linux-mars |
| ellipse-001 | KNOWN_TRUTH_EXACT | SupportedExact | 1 | 1 | 384 | 2.3 | x86_64-linux-mars |
| elliptical-arc-001 | KNOWN_TRUTH_EXACT | SupportedEquivalent | 1 | 1 | 384 | 2.4 | x86_64-linux-mars |
| empty-001 | QUARANTINED | NoGeometry | null | 0 | 0 | 0.8 | x86_64-linux-mars |
| general-affine-chain-001 | KNOWN_TRUTH_EXACT | SupportedEquivalent | 1 | 1 | 360 | 2.2 | x86_64-linux-mars |
| hatch-001 | QUARANTINED | NoGeometry | null | 1 | 0 | 0.7 | x86_64-linux-mars |
| hatch-gradient-001 | QUARANTINED | NoGeometry | null | 1 | 0 | 0.8 | x86_64-linux-mars |
| huge-extent-001 | KNOWN_TRUTH_EXACT | SupportedExact | 1 | 1 | 376 | 5.6 | x86_64-linux-mars |
| image-001 | QUARANTINED | NoGeometry | null | 1 | 0 | 0.9 | x86_64-linux-mars |
| large-coords-001 | KNOWN_TRUTH_EXACT | SupportedExact | 1 | 1 | 368 | 1.8 | x86_64-linux-mars |
| layer-state-001 | KNOWN_TRUTH_EXACT | SupportedEquivalent | 1 | 3 | 344 | 4.4 | x86_64-linux-mars |
| line-001 | KNOWN_TRUTH_EXACT | SupportedEquivalent | 1 | 1 | 352 | 9.9 | x86_64-linux-mars |
| lwpolyline-closed-001 | KNOWN_TRUTH_EXACT | SupportedExact | 1 | 1 | 368 | 2.4 | x86_64-linux-mars |
| mesh-001 | QUARANTINED | NoGeometry | null | 1 | 0 | 0.9 | x86_64-linux-mars |
| mixed-entity-001 | KNOWN_TRUTH_EXACT | NoGeometry | null | 8 | 0 | 0.8 | x86_64-linux-mars |
| mtext-001 | QUARANTINED | NoGeometry | null | 1 | 0 | 0.8 | x86_64-linux-mars |
| near-singular-001 | KNOWN_TRUTH_EXACT | NoGeometry | null | 0 | 0 | 0.7 | x86_64-linux-mars |
| nested-blocks-001 | KNOWN_TRUTH_EXACT | SupportedEquivalent | 1 | 1 | 352 | 2.6 | x86_64-linux-mars |
| nonuniform-scale-circle-001 | KNOWN_TRUTH_EXACT | SupportedExact | 1 | 1 | 360 | 2.3 | x86_64-linux-mars |
| nonuniform-scale-rotated-001 | KNOWN_TRUTH_EXACT | SupportedExact | 1 | 1 | 384 | 2.0 | x86_64-linux-mars |
| nonunit-weight-001 | KNOWN_TRUTH_EXACT | SupportedExact | 1 | 1 | 928 | 1.9 | x86_64-linux-mars |
| nurbs-d2-offconic-001 | KNOWN_TRUTH_EXACT | SupportedExact | 1 | 1 | 936 | 2.1 | x86_64-linux-mars |
| nurbs-d3-001 | KNOWN_TRUTH_EXACT | SupportedExact | 1 | 1 | 952 | 2.1 | x86_64-linux-mars |
| ole-001 | QUARANTINED | NoGeometry | null | 1 | 0 | 0.8 | x86_64-linux-mars |
| polyline-001 | KNOWN_TRUTH_EXACT | SupportedExact | 1 | 1 | 368 | 1.7 | x86_64-linux-mars |
| polyline2d-001 | KNOWN_TRUTH_EXACT | SupportedExact | 1 | 1 | 376 | 1.7 | x86_64-linux-mars |
| proxy-001 | QUARANTINED | NoGeometry | null | 0 | 0 | 0.6 | x86_64-linux-mars |
| reflection-001 | KNOWN_TRUTH_EXACT | SupportedEquivalent | 1 | 1 | 360 | 1.9 | x86_64-linux-mars |
| region-001 | QUARANTINED | NoGeometry | null | 1 | 0 | 0.7 | x86_64-linux-mars |
| repeated-knot-c0-001 | KNOWN_TRUTH_EXACT | SupportedExact | 1 | 1 | 920 | 2.0 | x86_64-linux-mars |
| rotation-irrational-001 | KNOWN_TRUTH_EXACT | SupportedEquivalent | 1 | 1 | 376 | 1.9 | x86_64-linux-mars |
| shear-chain-001 | KNOWN_TRUTH_EXACT | SupportedEquivalent | 1 | 1 | 352 | 2.0 | x86_64-linux-mars |
| singular-001 | KNOWN_TRUTH_EXACT | NoGeometry | null | 0 | 0 | 0.6 | x86_64-linux-mars |
| solid3d-001 | QUARANTINED | NoGeometry | null | 1 | 0 | 0.8 | x86_64-linux-mars |
| table-001 | QUARANTINED | NoGeometry | null | 1 | 0 | 0.9 | x86_64-linux-mars |
| text-001 | QUARANTINED | NoGeometry | null | 1 | 0 | 0.8 | x86_64-linux-mars |
| tiny-extent-001 | KNOWN_TRUTH_EXACT | SupportedEquivalent | 1 | 1 | 360 | 1.9 | x86_64-linux-mars |
| tiny-radius-001 | KNOWN_TRUTH_EXACT | SupportedExact | 1 | 1 | 368 | 1.8 | x86_64-linux-mars |
| translation-001 | KNOWN_TRUTH_EXACT | SupportedEquivalent | 1 | 1 | 352 | 1.7 | x86_64-linux-mars |
| uniform-scale-001 | KNOWN_TRUTH_EXACT | SupportedExact | 1 | 1 | 360 | 1.8 | x86_64-linux-mars |
| wipeout-001 | QUARANTINED | NoGeometry | null | 1 | 0 | 0.6 | x86_64-linux-mars |
| xref-001 | QUARANTINED | NoGeometry | null | 1 | 0 | 0.9 | x86_64-linux-mars |
| zero-length-segment-001 | QUARANTINED | NoGeometry | null | 1 | 0 | 0.6 | x86_64-linux-mars |
| zero-radius-001 | QUARANTINED | NoGeometry | null | 1 | 0 | 0.9 | x86_64-linux-mars |

## What the refusals are

The `QUARANTINED` fixtures are the entity classes the mapping does not carry yet (TEXT, MTEXT, HATCH, DIMENSION,
IMAGE, MESH, REGION, 3DSOLID and the rest), refused with their reason code. Two things in the table are not that and are
worth reading:

- `bulge-arc-001`, `bulge-arc-neg-001` and `mixed-entity-001` are `KNOWN_TRUTH_EXACT`. The classifier calls their
  entities supported, and then PVec's container refuses the curve with `WRT-08` ("a curve the container cannot store exactly
  and deterministically"), which fails the whole conversion instead of being counted against the one entity. That is a
  defect of the conversion path (#1446), and it is why these rows say `NoGeometry` with a refusal.
- `singular-001` and `near-singular-001` are refused at block expansion with `TRANSFORM_SINGULAR (NP-19)`, which is the
  expected outcome for a singular transform.
