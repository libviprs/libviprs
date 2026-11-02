# Vector pipeline: performance, precision and hardening report

This is the report #1036 asks for. It covers what the DWG to PMTiles tiler
costs on big and hostile input, what stops it from eating a worker, and where
the precision budget from #1032 stands. Every number below was measured, and the
command that produces each one is next to it, so a later change can be checked
against this file instead of against my memory of it.

Hardware and build for every timing: MARS (native x86_64, 24 cores, 32 GB),
`rust:1.99.1-bookworm` in a container, `--release` unless it says otherwise.
The tiler is single threaded, so core count does not move these.

## What a hostile drawing can do, and what stops it

| Attack | Before | Now |
|---|---|---|
| Two corner to corner lines at zoom 24 (a tile explosion) | 60.5 s, then an error, with 6,520 MiB live | stopped at zoom 20 after 6.8 s with 871 MiB live |
| 20,000 long lines at the default zoom 12 | 51.9 s, then an error, 1,896 MiB live | stopped at zoom 9 after 12.3 s with 644 MiB live |
| One random polygon of 2,000,000 vertices | 24.1 s, then an error, 1,799 MiB live | stopped at zoom 4 after 11.3 s with 834 MiB live |
| One 64 MiB label | 64 MiB copied per zoom into the tile | cut to 16 KiB before it is read whole, 0 MiB, 3.5 ms |
| Anything that runs too long | nothing stopped it | `TilerConfig::deadline` and `cancel`, polled inside a single primitive |
| Native decoder that never returns | nothing | the worker process's ceiling and wall clock (below) |

The old failure was the point budget being the only memory budget.
`max_level_points` counts points, and a line split over millions of tiles is
millions of two-point runs whose bookkeeping (a `Vec` header, an allocator
header, a map node) costs about ten times what the points do. `max_level_bytes`
(default 1 GiB) counts that bookkeeping, using `LEVEL_RUN_COST` (64 bytes) and
`LEVEL_TILE_COST` (160 bytes) per run and per tile. The estimate runs a little
high on purpose: the 1 GiB budget stopped runs whose measured peak was
644 to 871 MiB, never one over a gigabyte.

## Timing and memory, measured

`cargo test --release --test vector_1036_memory -- --ignored --nocapture performance_report`

Peak working heap is live heap the tiler added, measured by a counting global
allocator, after the input primitives were built (those are the caller's).

| case | primitives | time | peak working heap | outcome |
|---|---|---|---|---|
| 1M short lines, zoom 0..=6 | 1,000,000 | 1.97 s | 98 MiB | ok, 5,209 tiles |
| 1M short lines, zoom 0..=12 (default) | 1,000,000 | 12.14 s | 573 MiB | ok, 5,510,529 tiles |
| 20k long lines, zoom 0..=12 (default) | 20,000 | 12.34 s | 644 MiB | stopped by the level byte budget |
| 2 corner lines, zoom 0..=16 | 2 | 0.59 s | 74 MiB | ok, 756,125 tiles |
| 2 corner lines, zoom 0..=24 | 2 | 6.83 s | 871 MiB | stopped by the level byte budget |
| one 2M-vertex random polygon | 1 | 11.33 s | 834 MiB | stopped by the level byte budget |
| one 64 MiB label and a line | 2 | 3.5 ms | 0 MiB | ok, label cut and counted |
| 300k short labels, zoom 0..=12 | 300,000 | 3.68 s | 101 MiB | ok, 1,095,472 tiles |

Real drawings, through the decoder and the tiler (`vector_1034_fidelity`):
`g13_many_inserts` (one block inserted 30,000 times) is 67,467 tiles and
7.9 million points in 2.6 s with the largest tile 420,127 bytes, under the
500 KiB budget. `g13_dimension_deep`, the nesting stress drawing, ends in
about 0.26 s with the hardened provider limits on.

Two things in the table are not good news and I am not hiding them:

* The dense million-line case drops 916,022 runs for the point budget and
  leaves 17 tiles over the 500 KiB size budget (the largest is 1.5 MB). They
  are all at the lowest zooms, where a million lines land in a handful of
  tiles. Nothing is silent (`dropped_for_point_budget` and `oversized_count`
  say so), but those tiles are not faithful. Level of detail for the low zooms
  belongs with the release gate (#1038).
* The budgets turn a hostile input into an error, not into a degraded tile.
  A caller that wants a drawing with long lines at zoom 20 raises
  `max_level_bytes` or lowers `max_zoom`.

## Precision

Unchanged from #1032 and re-measured today. `cargo test --release --test
vector_1032_tiler -- --nocapture`.

Source coordinate error against the quantisation bound, extent 4096:

| zoom | unit (drawing units) | bound | measured vertex error | in tile units |
|---|---|---|---|---|
| 6 | 3.967e-3 | 2.805e-3 | 2.173e-3 | 0.548 |
| 10 | 2.480e-4 | 1.753e-4 | 1.048e-4 | 0.423 |
| 14 | 1.550e-5 | 1.096e-5 | 8.500e-6 | 0.548 |
| 18 | 9.686e-7 | 6.849e-7 | 6.577e-7 | 0.679 |
| 22 | 6.054e-8 | 4.281e-8 | 3.178e-8 | 0.525 |

Curve deviation (a circle), budget 1.457 tile units:

| zoom | deviation | in tile units |
|---|---|---|
| 2 | 4.611e-2 | 0.726 |
| 4 | 1.185e-2 | 0.747 |
| 6 | 3.197e-3 | 0.806 |
| 8 | 9.296e-4 | 0.937 |

Nothing in the hardening changed a coordinate: the cap on a label and the
budgets act before or after the geometry, and the 19 tests of #1032 still pass.

## What limits what

| Limit | Where | Default |
|---|---|---|
| primitives held from one decode | `TilerConfig::max_primitives` | 5,000,000 |
| bytes held for one zoom level | `TilerConfig::max_level_bytes` | 1 GiB (estimate) |
| points held for one zoom level | `TilerConfig::max_level_points` | 64 Mi |
| bytes of one label | `TilerConfig::max_text_bytes` | 16 KiB |
| points and text per tile | `max_points_per_tile`, `max_text_per_tile` | 400,000 and 4,000 |
| wall clock | `TilerConfig::deadline` | none |
| cancellation | `TilerConfig::cancel` | none |
| what the DWG reader accepts | `ProviderLimits::hardened()` | 256 MiB input, 10M entities, 1 MiB strings, 1M points a record, depth 64, 2 GiB output |

`deadline` and `cancel` are off by default because the library cannot know what
a caller's budget is. A worker sets both. They are polled once per primitive,
once per split of a primitive over its tiles, and once per tile, so a single
enormous primitive is stopped inside the primitive (the tile explosion test has
two primitives in it). The decoder behind `build_archive_from_view` cannot be
interrupted from here: the deadline takes effect at the first primitive the
decoder hands over after it passes. A decoder stuck inside native code and
handing over nothing is the case the process limits exist for.

`ProviderLimits::default()` sets none of its bounds. That is right for your own
drawing and wrong for an upload, so a worker uses `hardened()`.

## The worker is the sandbox

A static library is packaging, not a sandbox. `tests/vector_1036_worker.rs` runs
the conversion in a child process (this test binary again, one test, a mode in
the environment), the way a worker pool would, and checks what the parent sees:

* **Memory ceiling.** With every budget switched off (what a naive worker does)
  and `ulimit -v` at 256 MiB, a zoom 24 tile explosion dies and the parent
  carries on and serves the next job. Under a real cgroup the same child is
  OOM-killed: run with `docker run --memory=256m`, it exits 137 after
  42 s, `memory.events` shows `oom_kill 1`, and the destination file is not
  there. That run is a lane now: `tools/cgroup-worker-lane.sh` does it every night
  on MARS after the native cells (a test binary cannot make its own cgroup), and fails
  unless the exit is 137, docker reports OOMKilled and no destination exists.
* **Wall clock.** A child that never reads a clock (a stand in for a native
  decoder that does not come back) is killed by the supervisor at 1.5 s and
  gone by 10 s at the latest, with signal 9, and the destination is absent.
* **The library's own deadline.** With budgets off and a 500 ms deadline the
  child leaves by `DeadlineExceeded`, exit code 3, and leaves nothing in the
  directory.

A killed process cannot clean up. What a SIGKILL or an OOM kill leaves is the
staging pair `<destination>.tmp.data` and `<destination>.tmp.idx` beside the
destination, never the destination (the archive is renamed into place only when
whole, so a half archive never wears the real name). A worker sweeps those two
names after it reaps the child. The test asserts that nothing else is left.

## Panics, UB and the corpus

* **Extreme values** (`vector_1036_hardening`): coordinates of 1e300 and 1e-300,
  a radius of 1e300, bulges of 1e9, text at the edge of the range, control
  characters and MTEXT codes, 100,000 identical vertices, a huge spline, under
  three configurations. No panic, none over its time bound. A proptest of 96
  cases over arbitrary finite values adds to it.
* **A fuzz target**, `fuzz/fuzz_targets/fuzz_vector_tiler.rs`: 710,098 runs in
  241 s (`cargo fuzz run -s none fuzz_vector_tiler`, 3 GB rss limit, 20 s
  timeout), no crash, no timeout, no out of memory. The sanitizer was off
  (`-s none`) to keep the build time down, so this finds panics, aborts and
  hangs and does not find memory errors in safe code, which cannot have them
  anyway. The crate has no `unsafe` in the tiler.
* **Mutated drawings through the real decoder**
  (`vector_1036_dwg_corpus`, needs the native archive): 48 drawings, 40
  deterministic mutations each (bit flips, zeroed runs, truncation, a run
  copied over another), so 1,920 hostile files, each decoded and tiled inside a
  child process under the hardened limits. No child died of a signal or a
  panic. The decoder is not Rust, so a crash in it is a crash of the caller,
  which is why every one of them ran in a child.
* `ProviderLimits::max_entities` bites: the 30,000-primitive fixture with a
  bound of 300 delivers under a tenth of it.

What that corpus is not: every file is a mutation of something ACadSharp's own
writer produced, so it covers the structural paths the fixtures reach. It is
not a collection of real drawings from outside, and no sanitizer has run over
the native library. Those are in the follow-up, libviprs/libviprs#1256, which stays open.
