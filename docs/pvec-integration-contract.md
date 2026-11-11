# The PVec integration contract

This is the freeze of the libviprs to PVec integration (N8.7). Five things are pinned, the pin is
`tests/fixtures/pvec_integration_contract.json`, and `tests/pvec_integration_contract.rs` fails when
this page and the pin disagree, when the pin and the code disagree, and when a pin is blank. Nothing here is
frozen before its benchmark: the baselines it was frozen against are listed last and the test holds each of
them to its recorded hash.

## The five pins

**1. PVec: the revision, the container version and the capability set this build writes.** This build links
`pvec` at revision `e1a768ba53b0fc4d530fc8b528c04159299c6f10` and writes container version 1.0. The capabilities it
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
| `layers.no_plot` | ignore |
| `layers.off` | exclude |
| `nested_inserts.max_depth` | 64 |
| `space` | model |
| `viewport_clipping` | unsupported |
| `xrefs` | report-external |

**4. The accepted fixture corpus revision.** `r3-open` of `libviprs/pvec-gold-fixtures`, gold commit
`a185ad489967d76432990c8bc1688d5b6fafeb9c`, vendored under `tests/fixtures/pvec-gold/` and held to its own index by
`tests/common/pvec_gold.rs`.

**5. The provider compatibility range.** The default provider is `acadrust`, at revision
`f1e00c11f6371c8cfcff2c726620a51a45a2ad1b`, and it takes DWG versions AC1014 to AC1032. A drawing below AC1014 is
refused at conversion start. `docs/pvec-version-compatibility.md` has the table.

## The baselines it was frozen against

Each file must exist, be non-empty and have this sha256, so a baseline cannot move after the freeze without this
page and the pin moving with it.

| baseline | sha256 |
|---|---|
| `tests/fixtures/pvec_bench_corpus.json` | `97eca65c287aaea1c014489cd5cc02e685e17b7f26b3961837414317e18bc5e5` |
| `tests/fixtures/pvec_bench_baseline.json` | `84afe74abb0efebab67c58797e699dd2e77d10616019d2bcde08c291cafb5022` |
| `tests/fixtures/pvec_corpus_baseline.json` | `6bd4c944688bc202c886ac31bf5d18a009d53e7e9f3f56e34c81c25fcb5c509e` |

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
| peak memory | 2.66 GiB tracked heap, 3.07 GiB process peak when this was decided; 1,323 MiB tracked and 1,706 MiB process peak (arm64), 1,687 MiB (x64) after #1481 | 573 MiB working heap |
| shape of the memory | linear, about 2.8 KB per entity when this was decided and about 1.35 KB tracked after #1481, doubling the drawing doubles the peak | bounded by the level byte budget, a hostile drawing is stopped |
| output | 22.2 MB `.pvec` with the analytic curves and a report | 5,510,529 tiles |
| what a curve is | an exact or equivalent PVec curve (EPIC P's classes) | tessellated, display fidelity |

The two are not the same drawing, so I draw no conclusion about which is faster: no cell has both a PVec artifact
and an L derivative. What decides the outcome is the memory column. PVec's conversion holds the whole drawing until
the artifact is sealed, so it missed its own 1.5 GiB ceiling at a million entities by a factor of two. After #1481 (#1524, #1528, #1537) it is 1,323 MiB tracked and 1,706 MiB process peak (arm64), 1,687 MiB (x64) at 1,025,000 entities, which is still linear and still over the original 1.0 GiB and 1.5 GiB. The criterion is restated to 1.4 GiB tracked, 1.75 GiB RSS and at most 1,450 bytes an entity (`docs/pvec-benchmarks.md`), and the bounded-memory claim is withdrawn. The writer's
spill (pvec#235, #1314, #1417) exists now but does not move that peak, see the paragraph below. Making PVec the default now would make a drawing of that size need over
3 GiB where the L path stays under 600 MiB. EPIC L cannot carry what PVec exists for, the analytic geometry. So both
stay, and the caller picks the path that fits the drawing and the use. This is the outcome to revisit when the
bounded-memory benchmark has a strategy that passes its ceiling: the decision is this page and one
line of the pin, and the test will say when they disagree.

Since this was recorded the writer's spill has landed (#1314, #1417): a spilled conversion adds 24 bytes an entity in the
writer instead of 391 to 419 (`docs/adr/pvec-tiers-and-spool-rescope.md`). I have since re-run the bounded-memory benchmark with the spill (`docs/pvec-benchmarks.md`, #1064). The tracked peak and the peak RSS at 1,025,000 entities are within about 1% with and without it (2,725 and 3,149 MiB direct, 2,725 and 3,117 MiB spilled), at every budget I tried, because libviprs' own per-entity state is what sits in memory before the writer runs. So the figures above stand and the outcome stays "both ship, the caller chooses"... What would change it is a PVec writer that takes geometry as it is classified, or the report rows out of memory, both listed under "Not done" in `docs/pvec-benchmarks.md`. #1481 closes without either.
