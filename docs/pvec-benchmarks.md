# PVec conversion benchmarks

This is the procedure and the numbers for what a DWG to PVec conversion costs. It follows the
convention of `docs/pmtiles-benchmarks.md`: `#[ignore]`d harness tests run with `--ignored
--nocapture`, no `criterion`, no `benches/`, a counting allocator for the tracked figures and
`VmHWM` for the process peak. The six drawing classes every figure is measured on are pinned in
`tests/fixtures/pvec_bench_corpus.json` (#1429), and the committed baselines live in
`tests/fixtures/pvec_bench_baseline.json`.

## Rules every row follows

- Each figure is the median of three runs, with the min and max printed beside it. A cell that
  takes milliseconds is run in a batch inside each sample, so a sample is at least a quarter second.
  A row whose spread is over 20% is marked `unstable` and never gates.
- A run waits for a quiet host: the one-minute load average under the cpu count for three polls in
  a row. If that never happens the run refuses and prints the loads it saw, and it checks again at
  the end and discards the run if the host got busy. A number taken under load is not reported.
- Every row names its platform (`<arch>-<os>-<host>`, from `LIBVIPRS_BENCH_HOST`, which a run must
  set). x86_64 rows come from the native x64 host (MARS) and never from a local emulation.
- A row with a blank column is not published, the harness fails instead.

## Provider extraction (N8.2, #1433)

`tests/pvec_provider_bench.rs` measures the provider's own cost: open the drawing and resolve
model space through `BlockResolver::flatten` (every INSERT expanded, no provenance sink), and
nothing after that. Normalization is not fed, so the number is the provider's alone.

```text
LIBVIPRS_BENCH_HOST=mars cargo test --release --features pvec-convert \
    --test pvec_provider_bench -- --ignored --nocapture provider_bench
```

Set `LIBVIPRS_BENCH_BASELINE_OUT=path` to write the rows in the baseline's shape. Against the
committed baseline a run fails on more than 10% loss of entities per second or more than 15% growth
in the tracked allocation peak, on a stable row.

Columns: `entities` (flattened leaves), `entities_per_s` (median, min, max), `allocated_bytes` (all
bytes requested in one extraction), `tracked_peak` (peak of the live heap above where the run
started), `peak_rss_bytes` (process high-water mark, reset per cell), `block_frames` (block
instances opened), `advanced_curve_share` (splines over leaves).

Measured on MARS (x86_64-linux-mars, native, 28 cpus), release build, third full run, which is the
committed baseline:

| cell | entities | entities/s (median) | allocated bytes | tracked peak | peak RSS | block frames | advanced-curve share |
|---|---|---|---|---|---|---|---|
| simple | 1,000 | 395,377 | 9,258,183 | 7,718,017 | 58,736,640 | 0 | 0 |
| block_heavy | 22,440 | 389,585 | 68,651,353 | 15,244,354 | 74,412,032 | 11,220 | 0 |
| spline_heavy | 10,000 | 389,986 | 101,479,338 | 79,148,295 | 139,296,768 | 0 | 1 |
| large_industrial | 1,025,000 | 980,902 | 2,367,328,094 | 1,683,836,880 | 3,450,322,944 | 75,000 | 0 |

Three full runs on the same host disagreed with each other by more than any one run disagreed with
itself: `simple` came out at 455,451, 395,377 and a first unbatched 357,825 entities per second, so
the 10% regression gate has about that much noise under it on the smallest cell. The committed
baseline is the lowest of the batched runs for that reason. The other cells stayed within 9% of each other across the two batched runs.

The `large_industrial` row is the one to read before N8.4: extracting a million entities with no
normalization and no writer already holds a tracked peak of 1.57 GiB and a process peak of 3.2 GiB.
That is the provider's flattened leaves (each carries its own provenance stub), not the writer's.

### Advanced-curve overhead

Published as a ratio against the `simple` cell, not as an absolute claim: nanoseconds per entity on
`spline_heavy` over nanoseconds per entity on `simple` came out at 1.07 and 1.01 in the two batched
runs I took (0.82 in the first, unbatched run). A spline costs about what a line, arc or circle
costs to extract here: the provider reads the record and the cost is dominated by what surrounds it.

### Block recursion sweep

`block_heavy` at nesting depth 1 to 16, each point sized to about 2^16 instances in total so the
time per instance compares across depths (the walk is timed, the open is not):

| depth | instances | ns per instance |
|---|---|---|
| 1 | 65535 | 1475.3 |
| 2 | 65535 | 1926.6 |
| 3 | 65541 | 2537.2 |
| 4 | 65535 | 2747.9 |
| 5 | 65565 | 3538.8 |
| 6 | 65583 | 3928.3 |
| 7 | 65659 | 4358.7 |
| 8 | 65535 | 4950.7 |
| 9 | 65919 | 5564.3 |
| 10 | 66495 | 6250.8 |
| 11 | 67551 | 6876.5 |
| 12 | 69615 | 7373.8 |
| 13 | 73719 | 7863.3 |
| 14 | 81915 | 8989.4 |
| 15 | 98301 | 9915.5 |
| 16 | 65535 | 9953.2 |

Time per instance is not flat within 25%: it grows about linearly with depth, from 1.5 microseconds
at depth 1 to 10 microseconds at depth 16 (a factor of 6.7). What fits a cost linear in depth, from reading the code (I did not isolate it with a profile), is the
provenance stub every leaf carries. `AcadrustProvider` clones the instance path (`block_path: self.path.clone()`, in
`src/pvec/dwg_acadrust.rs`) into each entity it reports, and an `InstancePath` is a `Vec` of steps
that each own a block name, so each leaf costs a copy proportional to its depth, and
`InstancePath::child` copies the whole parent path again for every INSERT. I did not change the
provider to chase the number, that is out of scope here, and it is the first thing worth looking
at when N8.6 tunes.

## View and transform normalization (N8.3, #1436)

`tests/pvec_normalization_bench.rs` measures what libviprs does between the provider and PVec, one stage at a time, on `block_heavy`, `paper_space` and `large_industrial`, each at three visibility mixes. The stages, and the public call each one is:

- **visibility** (`visibility`): `ViewProfile::apply`, layer state and layout filtering.
- **records** (`records`): the plumbing that turns a kept leaf into a `Normalized2DGeometry`. It is not one of the four stages, I time it as a fifth line so the pass adds up.
- **transform** (`transform`): `push_inner_coefficients` and `promote`, with the seven non-singular N1.5 classes cycled over the entities.
- **bounds** (`bounds`): `bounds_inputs`, exact source bounds.
- **clipping** (`clip`): `apply_clip_policy`. The DWG path has no clip stage in v1 (model space only, no viewport clip), so this one is measured on a generated PDF page of 200 clips and 20,000 stroked lines, as its own pass.

```text
LIBVIPRS_BENCH_HOST=mars cargo test --release --features pvec-convert \
    --test pvec_normalization_bench -- --ignored --nocapture normalization_bench
```

The benchmark asserts its own outputs: the kept count equals an independent count from the layer names, an identity promotion leaves every bound within 4 ulps of what it was (see below), a singular matrix is refused for every entity, the stage times add up to the pass within 5% (the remainder is in the row as `unattributed`), and the 90% mix is not slower than the 0% mix. A stage that grew more than 10% in nanoseconds per entity against the baseline fails the run.

The mixes hide leaves on an off or frozen layer (`hidden` in the generator, 0, 50 or 90 percent of the leaves drawn), so the kept share is what the profile reports, not what the label says.

Measured on MARS (x86_64-linux-mars, native), release build, nanoseconds per entity (visibility per leaf met, the others per kept leaf; bounds per promoted entity), the slower of two runs, which is the committed baseline:

| cell | mix | leaves | kept | visibility | records | transform | bounds | pass (ms) | unattributed |
|---|---|---|---|---|---|---|---|---|---|
| block_heavy | 0% | 22,440 | 22,440 | 651 | 103 | 173 | 23 | 21.33 | 0.045% |
| block_heavy | 50% | 22,440 | 11,220 | 737 | 98 | 163 | 21 | 19.71 | 0.000% |
| block_heavy | 90% | 22,440 | 2,244 | 824 | 112 | 168 | 23 | 19.16 | 0.000% |
| paper_space | 0% | 2,000 | 2,000 | 134 | 94 | 161 | 22 | 0.82 | 0.011% |
| paper_space | 50% | 2,000 | 990 | 177 | 100 | 167 | 23 | 0.64 | 0.030% |
| paper_space | 90% | 2,000 | 196 | 183 | 102 | 170 | 24 | 0.42 | 0.015% |
| large_industrial | 0% | 1,025,000 | 1,025,000 | 371 | 135 | 197 | 49 | 772.01 | 0.083% |
| large_industrial | 50% | 1,025,000 | 462,536 | 421 | 154 | 202 | 52 | 621.35 | 0.104% |
| large_industrial | 90% | 1,025,000 | 92,516 | 373 | 166 | 181 | 49 | 419.03 | 0.122% |

Stage share of the pass at the 0% mix: block_heavy visibility 68%, records 11%, transform 18%, bounds 2%; paper_space visibility 33%, records 23%, transform 39%, bounds 5%; large_industrial visibility 49%, records 18%, transform 26%, bounds 7%.

Filtering pays for itself on every cell: the 90% mix finishes the pass faster than the 0% mix (block_heavy 19.2 against 21.3 ms, paper_space 0.42 against 0.82 ms, large_industrial 419 against 772 ms). Visibility is the biggest line on `block_heavy`, and I have not isolated why (my guess is that `ViewProfile::apply` walks the blocks itself, which is a second walk after the provider's).

### Transform classes

One row per class of the N1.5 matrix, over the first 200,000 kept leaves of `large_industrial` (lines, polylines, circles and arcs), composition and promotion only. The singular row is the cost of the refusal, not of a conversion: every one of the 200,000 entities was refused and none produced geometry.

| class | ns per entity | geometry | refused |
|---|---|---|---|
| identity | 167 | 200,000 | 0 |
| translation | 167 | 200,000 | 0 |
| rotation | 204 | 200,000 | 0 |
| uniform_scale | 167 | 200,000 | 0 |
| non_uniform_scale | 164 | 200,000 | 0 |
| reflection | 165 | 200,000 | 0 |
| shear | 219 | 200,000 | 0 |
| singular | 25 | 0 | 200,000 |

One thing the equality check turned up: an identity promotion is not a bit-for-bit passthrough. 9,948 of the 200,000 bounds came back a few units in the last place away from the box of the untouched curve (I did not chase which entity kinds), at most 4 ulps. That is far inside P1.2's tolerance, so the benchmark checks for at most 4 ulps and reports the count rather than demanding bit equality, and I did not change PVec's behaviour to hide it.

### Clipping

The `pdf_clip` row: `apply_clip_policy` with the analytic bounds source over 20,200 paths and 200 clips: 52 ns per painted path.
