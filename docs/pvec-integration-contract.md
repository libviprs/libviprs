# The PVec integration contract

This is the freeze of the libviprs to PVec integration (N8.7). Five things are pinned, the pin is
`tests/fixtures/pvec_integration_contract.json`, and `tests/pvec_integration_contract.rs` fails when
this page and the pin disagree, when the pin and the code disagree, and when a pin is blank. Nothing here is
frozen before its benchmark: the baselines it was frozen against are listed last and the test holds each of
them to its recorded hash.

## The five pins

**1. PVec: the revision, the container version and the capability set this build writes.** This build links
`pvec` at revision `ff67ecb85b5367c5db56f7d03f162c02437414a8` and writes container version 1.0. The capabilities it
can write are the ones the mapping table (`pvec::mapping::NORMALIZED_TO_PVEC`) gives its ten kinds, ids 5 to 12 of
PVec's registry. What a build does with other container versions is `docs/pvec-version-compatibility.md`.

**2. The numeric policy, by reference and version.** `pvec_numeric::Policy::V1`. libviprs defines no epsilon of
its own: every tolerance a libviprs test or module uses is one of the five terms of that policy (`abs` 1e-12,
`rel` 1e-10, `ulp` 8, `weight_floor` 1e-8, `axis_ratio_max` 1e6), read from PVec, never copied. The test compares
the pin with the policy PVec exports, so a change of PVec's policy is a red test here and not a silent change.

**3. The view profile.** The v1 "visible flattened drawing" is `ViewProfile::default_profile()`, knob by knob:

| field | value |
|---|---|
| `layers.frozen` | exclude |
| `layers.frozen_in_viewport` | ignore-and-report |
| `layers.insert_override` | inherit-on-layer-zero |
| `layers.off` | exclude |
| `nested_inserts.max_depth` | 64 |
| `space` | model |
| `viewport_clipping` | unsupported |
| `xrefs` | report-external |

**4. The accepted fixture corpus revision.** `r2-open` of `libviprs/pvec-gold-fixtures`, gold commit
`1df9d1b3e75047fa3978ebd809d0128cdfb28a68`, vendored under `tests/fixtures/pvec-gold/` and held to its own index by
`tests/common/pvec_gold.rs`.

**5. The provider compatibility range.** The default provider is `acadrust`, at revision
`f1e00c11f6371c8cfcff2c726620a51a45a2ad1b`, and it takes DWG versions AC1014 to AC1032. A drawing below AC1014 is
refused at conversion start. `docs/pvec-version-compatibility.md` has the table.

## The baselines it was frozen against

Each file must exist, be non-empty and have this sha256, so a baseline cannot move after the freeze without this
page and the pin moving with it.

| baseline | sha256 |
|---|---|
| `tests/fixtures/pvec_bench_corpus.json` | `781b90f9cdebb44df472aadbfdff7739dc8e1e3468c320e49b2fd902cfb43622` |
| `tests/fixtures/pvec_bench_baseline.json` | `130d5f80fa1589b43a2780e48a063fbcd665fe8df8738d377a511fe964ba3e82` |
| `tests/fixtures/pvec_corpus_baseline.json` | `85cea1e0a1cfa6ef547368cb2d44ab71c1ed641bc8aca13694b5f29ac2d01099` |

## The EPIC L decision

EPIC L (#1028) ships the tessellated DWG to MVT to PMTiles path and this freeze does not touch it: the diff shows
`src/vector/` and its documents as they were, and the test checks that the paths named in the pin are still there.
Whether PVec becomes the default blueprint path is recorded here, with the numbers beside it. Three outcomes are
admissible: PVec becomes the default, EPIC L remains the default, or both ship and the caller chooses.

**Outcome: both ship and the caller chooses.**

The runs that informed it (cell, date, platform):

- provider/large_industrial, 2026-11-01, x86_64-linux-mars
- bounded_memory/direct/large_industrial/x1.0, 2026-11-01, x86_64-linux-mars
- generation/large_industrial, 2026-11-01, x86_64-linux-mars
- vector tiler, 1M short lines zoom 0..=12 (docs/vector-performance-report.md), 2026-10-31, x86_64-linux-mars

What the numbers say, in `docs/pvec-benchmarks.md` for the PVec side and `docs/vector-performance-report.md` for the L side:

| | PVec (1,025,000 entities, `large_industrial`) | EPIC L (1,000,000 short lines, zoom 0 to 12) |
|---|---|---|
| conversion wall time | 8.2 s end to end (1.05 s of it the provider, 1.0 s the writer) | 12.1 s |
| peak memory | 2.66 GiB tracked heap, 3.07 GiB process peak | 573 MiB working heap |
| shape of the memory | linear, about 2.8 KB per entity, doubling the drawing doubles the peak | bounded by the level byte budget, a hostile drawing is stopped |
| output | 22.2 MB `.pvec` with the analytic curves and a report | 5,510,529 tiles |
| what a curve is | an exact or equivalent PVec curve (EPIC P's classes) | tessellated, display fidelity |

The two are not the same drawing, so I draw no conclusion about which is faster: no cell has both a PVec artifact
and an L derivative. What decides the outcome is the memory column. PVec's conversion holds the whole drawing until
the artifact is sealed, so it misses its own 1.5 GiB ceiling at a million entities by a factor of two, and the spool
that would bound it is not built (pvec#235). Making PVec the default now would make a drawing of that size need over
3 GiB where the L path stays under 600 MiB. EPIC L cannot carry what PVec exists for, the analytic geometry. So both
stay, and the caller picks the path that fits the drawing and the use. This is the outcome to revisit when the spool
lands and the bounded-memory benchmark has a strategy that passes its ceiling: the decision is this page and one
line of the pin, and the test will say when they disagree.
