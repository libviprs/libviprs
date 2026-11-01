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
