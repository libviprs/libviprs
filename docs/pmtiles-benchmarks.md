# PMTiles benchmarks, bounded memory, and release readiness

This is the procedure behind the numbers libviprs.org publishes for PMTiles
storage, and the checklist that says PMTiles is ready to be the default. It
covers what is measured, how to rerun it, what each exported column means, and
what the bounded-memory proof does and does not claim.

Everything here runs out of this repository. There is no `criterion` and no
`benches/` directory, and there is a reason rather than an omission: the
interesting numbers are peak RSS, output size and filesystem-entry count, which
are properties of one run rather than of a sample of many, and the read numbers
want caches that repeated iterations destroy by construction. So the harness
reuses the `#[ignore]`d wall-clock convention the crate already uses and the
`MemoryTracker` the engine already reports through `EngineResult`.

## What gets measured

Two backends, over the same source and the same plan:

- the directory backend, `FsSink` writing loose PNG tiles under the layout's own
  tile paths, read back through `DirectoryPyramidReader`
- PMTiles, `PmTilesSink` writing one indexed v3 archive, read back through
  `PmTilesPyramidReader`

Generation reports wall time, tiles per second, the engine's tracked working
set, process peak RSS, the bytes the pyramid occupies and the number of
filesystem entries it occupies. That last column is the one the whole epic is
about: an archive is one entry however many tiles it holds, and a tree is one
entry per tile plus every level and column directory above it.

Reads report five scenarios. `read_cold` gives every lookup a reader that has
never been used, `read_warm` walks the same coordinates on one reader that has
already seen them, `read_sequential` walks the plan in order, `read_random`
walks a deterministic shuffle, and `read_concurrent` runs the same shuffle
across a ladder of thread counts.

### The cold row, split

`read_cold` is one number over five different pieces of work, and issue #1021
is about the fact that a fix would be different for each of them. So the same
cold open is measured a second time with each step timed on its own, and the
six phases are published alongside the combined row rather than instead of it,
so the history stays comparable:

| Phase | What it is |
|---|---|
| `read_cold_open` | opening the file and asking its size |
| `read_cold_header` | the 127-byte header read and decode |
| `read_cold_root_fetch` | the ranged read of the compressed root |
| `read_cold_root_inflate` | inflating it, capped at `MAX_DIRECTORY_BYTES` |
| `read_cold_root_decode` | `deserialize_entries` and its four varint passes |
| `read_cold_lookup` | the binary search and the one `pread` for the tile |

The phases are walked by hand through the same public API in the same order
`Reader::try_new` uses, rather than by instrumenting the reader, because #1021
is a measurement issue and nothing on the product's hot path should change to
be measured. What makes that honest is that the phases have to be the same work:
`the_cold_split_accounts_for_the_whole_combined_row` runs the split's own steps
and a real cold open side by side, over a source that records every byte range
it is asked for, and fails unless the split's timed phases read exactly what the
open reads, decode the header the reader goes on to use, rebuild the reader's
root directory and return the reader's tile. A split that does not reconcile is
measuring something else, and it would look exactly as plausible on a chart.

Only `read_cold_lookup` fetches a tile, so it is the only phase with
`tile_bytes_returned`. The other five publish `null` there rather than `0`,
which would read as a phase that fetched a tile for free.

Only PMTiles is split. The directory backend's open is one `is_dir()` stat with
no header, no ranged read and no index to decode, so splitting it would give
four rows of nothing and one that is the whole cost. That asymmetry is the
finding, and the combined row already carries it.

The cold row times the open as well as the lookup. For PMTiles that is a header
fetch and a root-directory fetch, which is what a client really pays before its
first tile; for the directory backend opening reads nothing at all, because a
tree has nothing to read up front. Leaving it out would price one backend for
work the other does not do.

**Cold means a cold reader, not a cold page cache.** Dropping the OS page cache
needs root on Linux and has no portable equivalent, so the harness does not
claim to have done it. The gap between the cold and warm rows is the open cost
plus the in-process caches (the PMTiles leaf cache, the directory reader's lack
of one), which is the part libviprs controls and the part an optimisation would
move.

One more thing the read rows are not, so nobody reads more into them than they
hold. `read_random` runs on the reader `read_sequential` has just walked end to
end, so its leaf cache is whatever that pass left behind rather than empty; the
row is a warm random walk and it is compared against a warm sequential one.

### The thread ladder

`read_concurrent` runs at 1, 2, 4 and 8 threads and publishes a row for each,
with the thread count in `concurrency`. **One thread is the control.** It used
to be a single row at whatever `available_parallelism` reported, and a slow row
there could have been contention or could have been per-lookup cost that was
present at every width, with nothing in the export able to separate them. The
shape of the curve against its own T=1 point is what answers that.

Eight threads run even on a box with fewer cores. That is oversubscription
rather than parallelism and it is left in deliberately, because the envelope's
provenance records `ncpu` and a reader can see which points on the curve had a
core to themselves.

### Root-only archives, the cell that is not, and the cell at the brink

Under 16384 directory entries the writer puts the whole directory in the root,
so an archive of a few thousand tiles never exercises the leaf lookup, the leaf
cache or the second ranged read. Three of the five cells in the large profile
are in that regime and so is the CI cell.

Two notes on that number. The comparison in `build_directories` is strict
(`plan.entry_count < ROOT_ONLY_MAX_ENTRIES`), so the largest flat root this
writer emits holds **16383** entries and not 16384. And `entry_count` counts
run-length-encoded **entries**, not tiles: neighbouring tiles that share a
payload collapse into one entry, so a deduplicating pyramid has far fewer
entries than tiles. The benchmark's source is a gradient, whose tiles are all
distinct, so for these cells the two numbers are equal, and the brink cell's
test asserts that rather than assuming it.

The fourth is 8192 pixels at a **64 pixel** tile: 21851 tiles, past the cutoff,
so its archive really has leaf directories. Reaching that through the tile size
rather than through a bigger canvas is deliberate, because that many tiles at
256 pixel tiles needs a 32768 pixel source and that is a 3.2 GB raster, while
what the read path cares about is the directory shape rather than the pixels
behind it. `the_large_profile_reaches_the_leaf_directory_path` asserts the sweep
still crosses the cutoff, so a future edit to the cell list cannot quietly drop
the only cell that covers half the read path.

The count is 21851 and not 21845, and both numbers are real, which is why this
document carried one in its prose and the other in its tables until I checked.
21845 is the sum of the eight levels whose source is at least one tile across,
which is the number most people mean by a full pyramid at 8192 pixels and 64
pixel tiles. The planner keeps halving the source until it is one pixel, so
there are fourteen levels rather than eight and the last six carry one tile
each. `the_eight_thousand_pixel_cell_plans_the_tile_count_the_doc_publishes`
pins both numbers and the level count, so the next person does not have to
derive it again.

The fifth cell is the brink: 4096 by 6256 pixels at a **46 pixel** tile, which
plans 16369 tiles and comes out as a flat root of 16369 entries, 14 under the
16383 the writer will still keep flat. It is there because the other four sit
at 93, 1373, 5469 and 6 root entries, so the sweep bracketed the worst case
without ever touching it and issue #1021's 277 us peak was a line fitted
through three points. Measured, that open costs 245 us on the arm64 box below
and 320 on the x86_64 one, so the extrapolation was the right order and the
wrong number, and the reason it was the wrong number is in the phase split.
The shape looks arbitrary because it is a search result rather than a choice: `the_brink_cell_is_the_largest_root_the_search_space_reaches`
re-runs that search over tile sizes from 16 pixels and canvases up to 4096, and
fails if some other cell gets closer.

What pins it is not the arithmetic, though.
`the_brink_cells_root_stops_just_under_the_writers_cutoff` builds the archive,
opens it and asks `root_entries()`, because a cell pinned by arithmetic over
the writer's own halving silently stops being the brink cell the day the writer
changes. It is `#[ignore]`d, since it writes a fifty megabyte archive:

```sh
cargo test --release --test pmtiles_benchmarks -- --ignored --nocapture \
  the_brink_cells_root_stops_just_under_the_writers_cutoff
```

## Running it

The cheap profile is the default and takes seconds:

```sh
cargo test --release --test pmtiles_benchmarks -- --ignored --nocapture
```

The large profile walks five cells, up to 16384x16384 pixels and up to 21851
tiles, and takes minutes. It is opt-in on purpose, since a benchmark nobody runs
because it is too expensive is a benchmark nobody runs:

```sh
LIBVIPRS_BENCH_PROFILE=large \
LIBVIPRS_BENCH_JSON=/tmp/pmtiles-benchmarks.json \
cargo test --release --test pmtiles_benchmarks -- --ignored --nocapture
```

`LIBVIPRS_BENCH_PROFILE` selects `ci` (the default) or `large`.
`LIBVIPRS_BENCH_JSON` chooses where the export lands; without it the export
goes to `target/pmtiles_results.json`. It is deliberately not named
`scalability_results.json`: that file is a generated artefact of libviprs-bench
and a hand-placed file on its name is a number nobody can trace back to a run.

Run it in the Linux container rather than on a developer machine if the peak
RSS column matters. It comes from `/proc/self/status`, which exists on Linux
and nowhere else this crate builds for, and a platform with no answer publishes
`null` rather than a number invented to fill the column.

### Every measurement is a fresh process

Peak RSS is a high-water mark the kernel never lowers on its own, so two
backends measured in one process hand the second one the first one's peak. The
benchmark test re-executes its own test binary once per cell and each cell
reports its own process's numbers; inside a cell, the read phase resets the
high-water mark through `/proc/self/clear_refs` so generation's peak does not
become the read scenario's floor.

## The exported JSON

An envelope, `{"schema": 2, "provenance": {...}, "rows": [...]}`, written to the
path `LIBVIPRS_BENCH_JSON` names. The envelope is there so a consumer can refuse
a document it was not written against instead of reading a renamed column as
absent, and `schema` is the number that changes when a field changes meaning.

Schema 2 is issue #1021's: the envelope gained `provenance`, rows gained
`root_entries`, `scenario` gained the six `read_cold_*` phases, and
`read_concurrent` became one row per thread count. All of it is additive, so a
schema 1 consumer reading by name still finds every column it knew, but it
would plot the phase rows as though they were whole cold opens, which is why
the number moved rather than staying put.

### Where the numbers came from

```json
"provenance": {
  "commit": "d596675",
  "dirty": false,
  "rustc_version": "rustc 1.98.1 (48a229cea 2026-09-01)",
  "build_profile": "release",
  "host": {
    "cpu_model": null,
    "ncpu": 8,
    "arch": "aarch64",
    "os": "linux",
    "in_container": true
  },
  "load_average": { "one_min": 0.22, "five_min": 1.47, "fifteen_min": 2.98 }
}
```

That one is the arm64 run this document publishes. `cpu_model` is `null` there
and `"Intel(R) Pentium(R) Gold 8505"` on the x86_64 run, for the reason below.

A wall-clock number with no host attached is not a measurement anybody else can
check. The published figures before this carried none of it, and their own
prose said "amd64 container" on an Apple Silicon machine, which means Rosetta
and nothing in the document could have told a reader that. So the parent
process samples the host once, before any cell runs, and prints a warning to
stderr for every condition that spoils the numbers: a debug build, a one-minute
load average at or above the CPU count, a dirty tree whose commit therefore
describes nothing, or no commit at all.

`arch` is the field that answers the emulation question and it is a
compile-time fact, so it cannot be wrong. `cpu_model` is `null` on aarch64
Linux, where `/proc/cpuinfo` carries no `model name` line, rather than a name
invented to fill it. `commit` is `null` when git cannot answer, and the case
that turns up in practice is a linked git worktree bind-mounted into a
container: its `.git` is a file pointing at a gitdir outside the mount, so a
run from an agent worktree publishes a null commit while a run from a clone
publishes a real one.

The shape mirrors `libviprs_bench::provenance::Provenance`, which solved this
first (its issue #159) and already had the load-average warning. It is copied
rather than imported, because `libviprs-bench` depends on this crate and a
benchmark harness is the last place to take a dependency for seven fields.

`engine` says which engine produced the row and is `"libviprs"` on every row
here, because both sides of this comparison run the same one. `storage` says
which backend, and that is what varies. They used to hold the same string,
which made one of them a duplicate under a name that says something else.

**A column a row did not measure is `null`, never `0`.** A generation row
measures the pyramid it wrote and no latencies; a read row measures latencies
and not the pyramid, which the generation row for the same pyramid already did.
A zero would be a value on a scale somebody plots: `filesystem_entries: 0` reads
as better than the `1` a real archive costs, and `resource_cost: 0` is the best
possible score on a column where lower is better. Both of those were what a
failed measurement used to publish.

| Column | Meaning |
|---|---|
| `width` | Source canvas width in pixels |
| `height` | Source canvas height in pixels |
| `megapixels` | `width * height / 1e6` |
| `tile_size` | Tile edge in pixels. Two rows at one canvas and two tile sizes are not the same measurement: the tile count, and so the archive's directory shape, follows from it |
| `engine` | The engine under test. `"libviprs"` on every row here |
| `concurrency` | Threads the row was measured at. 1 everywhere except `read_concurrent`, which publishes a row at each of 1, 2, 4 and 8 |
| `wall_time_ms` | Wall-clock milliseconds for the whole row |
| `tracked_memory_mb` | The engine's own `MemoryTracker` peak: raster buffers, nothing else. `null` on a read row, which allocates none |
| `peak_rss_mb` | Process peak resident set for this phase. `null` where the platform has no answer |
| `tiles_produced` | Tiles written, or tiles read back for a read row |
| `tiles_per_second` | `tiles_produced` over wall time. `null` when the row took no measurable time |
| `tiles_per_second_per_mb` | Throughput per peak-RSS megabyte, higher is better. `null` whenever `peak_rss_mb` is |
| `resource_cost` | RSS-megabyte-seconds per tile, lower is better. `null` whenever `peak_rss_mb` is |
| `scenario` | `generate`, `read_cold`, `read_warm`, `read_sequential`, `read_random`, `read_concurrent`, or one of the six `read_cold_*` phases the cold row splits into |
| `storage` | `pmtiles` or `directory`, the backend the row measured |
| `profile` | `ci` or `large` |
| `output_bytes` | Bytes the pyramid occupies on disk. `null` on a read row, and `null` when the path could not be walked |
| `filesystem_entries` | Filesystem entries it occupies, directories included. 1 for an archive. `null` on a read row, and `null` when the path could not be walked |
| `root_entries` | Entries in the archive's root directory, as the archive answers it. This is the x axis of the cold-open ramp: an open decodes the whole root, so what a first lookup costs follows from this and not from the tile count or the file size. Run-length-encoded entries, not tiles. `null` on every directory row, because a tree has no root to decode |
| `tile_bytes_returned` | Tile payload bytes the row's lookups returned, summed. `null` on a generation row |
| `p50_latency_us` | Median per-lookup latency in microseconds. `null` on a generation row |
| `p99_latency_us` | 99th-percentile per-lookup latency in microseconds. `null` on a generation row |

`tile_bytes_returned` was `bytes_fetched`, and the rename is the point of it.
It sums the lengths of the payloads the lookups handed back, which is not bytes
off the transport, and bytes off the transport is exactly what the index-only
proof is about. A reader would have taken that column as evidence for a claim it
does not measure. Transport bytes are counted in
`tests/pmtiles_index_only_reads.rs`, against a `RangeReader` that can see them.

The two ratio columns are derived the way the existing scalability producer
derives them, both on the peak-RSS basis, so a row here means the same thing as
a row there. The one difference is the missing denominator: that producer
answers `0` and this one answers `null`.

## The numbers, as measured

Everything below was taken twice on each of two machines, at commit `d596675`,
release profile, in a container, on the container's own filesystem, over a
gradient source with `Layout::Xyz` and PNG tiles. The two numbers in every cell
are the two runs.

| | native arm64 | native x86_64 |
|---|---|---|
| CPU | Apple M5 | Intel Pentium Gold 8505, one P core and four E cores |
| Kernel | Docker Desktop's Linux VM | Linux 6.12.30, on the metal |
| CPUs the container saw | 8 | 6 |
| `rustc` | 1.98.1 | 1.98.1 |
| One-minute load when sampled | 0.22 and 0.45 | 0.23 and 0.47 |

**Neither run is emulated, and saying so is the point of that table.** The
figures this section replaces were taken in an amd64 container on an Apple
Silicon Mac, which is Rosetta, and nothing in the document could have told a
reader, because the document had nowhere to record it. Every table from here on
names its architecture, and the exported JSON carries the rest.

The two machines are not peers. Depending on the row the M5 is between about one
and a half and three times quicker per core, and the 8505's four E cores are not
the same core as its one P core. So what is worth reading across the two columns is shape rather than
size: whether the ramp has the same slope, whether the thread curve knees in the
same place, whether the sign of a comparison flips.

### Generation, native arm64

| Cell | Tiles | Backend | Wall time | Tiles/s | Peak RSS | Output bytes | Filesystem entries |
|---|---|---|---|---|---|---|---|
| 2048x2048 @256 | 93 | directory | 52.7 / 50.8 ms | 1764.3 / 1832.0 | 30.6 / 30.6 MB | 8 147 032 | 129 |
| 2048x2048 @256 | 93 | pmtiles | 67.6 / 70.0 ms | 1376.0 / 1329.3 | 30.6 / 30.6 MB | 8 147 559 | **1** |
| 8192x8192 @256 | 1373 | directory | 751.2 / 695.5 ms | 1827.7 / 1974.2 | 435.5 / 435.5 MB | 130 159 233 | 1459 |
| 8192x8192 @256 | 1373 | pmtiles | 959.2 / 1010.8 ms | 1431.3 / 1358.3 | 435.7 / 435.7 MB | 130 160 332 | **1** |
| 16384x16384 @256 | 5469 | directory | 3748.4 / 3082.1 ms | 1459.0 / 1774.4 | 1731.5 / 1731.5 MB | 520 598 291 | 5620 |
| 16384x16384 @256 | 5469 | pmtiles | 4018.2 / 4168.4 ms | 1361.1 / 1312.0 | 1732.3 / 1732.3 MB | 520 600 896 | **1** |
| 8192x8192 @64 | 21851 | directory | 1301.1 / 1271.6 ms | 16794.6 / 17184.5 | 435.2 / 435.1 MB | 133 748 027 | 22127 |
| 8192x8192 @64 | 21851 | pmtiles | 1159.3 / 1147.5 ms | 18849.2 / 19042.1 | 437.1 / 437.1 MB | 133 758 824 | **1** |
| 4096x6256 @46 | 16369 | directory | 522.4 / 502.2 ms | 31331.7 / 32595.4 | 168.1 / 168.1 MB | 52 318 406 | 16572 |
| 4096x6256 @46 | 16369 | pmtiles | 498.8 / 529.9 ms | 32817.4 / 30889.7 | 169.3 / 169.2 MB | 52 326 948 | **1** |

### Generation, native x86_64

| Cell | Tiles | Backend | Wall time | Tiles/s | Peak RSS | Output bytes | Filesystem entries |
|---|---|---|---|---|---|---|---|
| 2048x2048 @256 | 93 | directory | 94.5 / 95.3 ms | 984.2 / 976.2 | 31.1 / 31.0 MB | 8 147 032 | 129 |
| 2048x2048 @256 | 93 | pmtiles | 265.8 / 270.7 ms | 349.9 / 343.5 | 31.0 / 30.9 MB | 8 147 559 | **1** |
| 8192x8192 @256 | 1373 | directory | 1477.1 / 1465.7 ms | 929.5 / 936.8 | 435.8 / 435.9 MB | 130 159 233 | 1459 |
| 8192x8192 @256 | 1373 | pmtiles | 2787.2 / 2748.8 ms | 492.6 / 499.5 | 436.4 / 436.4 MB | 130 160 332 | **1** |
| 16384x16384 @256 | 5469 | directory | 5981.5 / 5815.7 ms | 914.3 / 940.4 | 1731.8 / 1732.1 MB | 520 598 291 | 5620 |
| 16384x16384 @256 | 5469 | pmtiles | 10634.2 / 11246.9 ms | 514.3 / 486.3 | 1732.5 / 1732.6 MB | 520 600 896 | **1** |
| 8192x8192 @64 | 21851 | directory | 2021.5 / 2131.3 ms | 10809.5 / 10252.5 | 435.6 / 435.4 MB | 133 748 027 | 22127 |
| 8192x8192 @64 | 21851 | pmtiles | 2996.8 / 2984.7 ms | 7291.4 / 7321.0 | 437.7 / 437.6 MB | 133 758 824 | **1** |
| 4096x6256 @46 | 16369 | directory | 1084.4 / 1011.3 ms | 15094.7 / 16186.4 | 168.5 / 168.5 MB | 52 318 406 | 16572 |
| 4096x6256 @46 | 16369 | pmtiles | 1261.3 / 1312.5 ms | 12978.2 / 12471.2 | 169.7 / 169.9 MB | 52 326 948 | **1** |

The engine's tracked working set is identical for the two backends at every cell
on both machines, which it should be: it charges raster buffers and a sink is
not one. Output bytes and filesystem entries are identical across architectures
too, because they are properties of the format and not of the box.

The last column is the one the whole epic is about, and nothing moves it. 22127
filesystem entries against 1, for the same 133 MB of tiles.

### What generation actually costs, and what this document used to claim

It used to say PMTiles generation costs between 1% and 31% more wall time, and
that at 21851 tiles of 6 KB the two backends are within 1%. Neither sentence
survives measuring it on two machines:

| Cell | Tiles | arm64 pmtiles/directory | x86_64 pmtiles/directory |
|---|---|---|---|
| 2048x2048 @256 | 93 | 1.28x / 1.38x | 2.81x / 2.84x |
| 8192x8192 @256 | 1373 | 1.28x / 1.45x | 1.89x / 1.88x |
| 16384x16384 @256 | 5469 | 1.07x / 1.35x | 1.78x / 1.93x |
| 8192x8192 @64 | 21851 | **0.89x / 0.90x** | **1.48x / 1.40x** |
| 4096x6256 @46 | 16369 | 0.95x / 1.06x | 1.16x / 1.30x |

At 21851 tiles the archive is about 10% **faster** than the tree on the M5 and
about 44% slower on the 8505. "Within 1%" was one machine's number written down
as though it were the crate's, and issue #1021 was right to say it did not
reproduce.

What does hold on both machines is the direction. The ratio falls as the tiles
get smaller, and the reason is structural rather than an oversight: the root
directory has to fit the spec's first 16 KiB and an entry's offset is not known
until the tiles are sorted, so the payloads are staged and then copied into the
archive, which is roughly twice the archive's size in writes for the same tiles.
That fixed copy matters less and less against the per-file cost of a tree as the
files get smaller and more numerous, and on a machine whose storage is quick
relative to its CPU it crosses over.

### Reads, native arm64

Microseconds per lookup. The lookup count is 64 for the cold and warm rows and
the whole plan (capped at 20000) for the rest.

| Cell | Scenario | Directory p50 | PMTiles p50 | Directory p99 | PMTiles p99 |
|---|---|---|---|---|---|
| 2048x2048 @256 | cold | 6.96 / 6.83 | 12.96 / 10.92 | 15.5 / 9.4 | 21.0 / 14.2 |
| 2048x2048 @256 | warm | 3.96 / 4.12 | **4.21 / 3.88** | 9.1 / 5.0 | 13.0 / 5.2 |
| 2048x2048 @256 | sequential | 4.21 / 4.29 | 4.17 / 4.62 | 15.9 / 7.3 | 10.1 / 9.7 |
| 2048x2048 @256 | random | 4.17 / 4.12 | 3.92 / 4.25 | 5.1 / 11.6 | 4.9 / 26.6 |
| 8192x8192 @256 | cold | 8.83 / 6.71 | 39.96 / 39.00 | 20.5 / 8.0 | 50.8 / 101.5 |
| 8192x8192 @256 | warm | 5.08 / 4.54 | **4.54 / 5.42** | 21.1 / 5.0 | 9.8 / 6.8 |
| 8192x8192 @256 | sequential | 7.21 / 6.33 | 8.17 / 9.29 | 15.4 / 11.7 | 16.8 / 18.7 |
| 8192x8192 @256 | random | 6.92 / 5.42 | 7.67 / 7.38 | 15.9 / 10.0 | 15.8 / 13.3 |
| 16384x16384 @256 | cold | 7.92 / 7.58 | 92.79 / 92.96 | 26.6 / 16.8 | 121.6 / 153.9 |
| 16384x16384 @256 | warm | 4.46 / 4.46 | 4.75 / 5.17 | 6.0 / 5.0 | 9.4 / 7.0 |
| 16384x16384 @256 | sequential | 7.04 / 7.17 | 7.71 / 8.58 | 14.2 / 18.5 | 14.3 / 28.0 |
| 16384x16384 @256 | random | 6.58 / 6.33 | 7.67 / 9.04 | 12.6 / 11.5 | 13.2 / 16.4 |
| 8192x8192 @64 | cold | 3.08 / 3.17 | 69.83 / 68.75 | 10.2 / 17.0 | 104.5 / 95.0 |
| 8192x8192 @64 | warm | 1.54 / 1.54 | **0.46 / 0.42** | 1.6 / 1.7 | 0.7 / 0.5 |
| 8192x8192 @64 | sequential | 3.08 / 3.04 | **1.08 / 1.08** | 7.2 / 5.0 | 1.9 / 1.9 |
| 8192x8192 @64 | random | 3.33 / 3.17 | **1.29 / 1.17** | 6.4 / 4.9 | 2.1 / 1.8 |
| 4096x6256 @46 | cold | 2.83 / 2.83 | 245.29 / 243.50 | 11.6 / 10.0 | 288.4 / 274.0 |
| 4096x6256 @46 | warm | 1.58 / 1.54 | **0.33 / 0.33** | 1.8 / 2.0 | 0.4 / 0.4 |
| 4096x6256 @46 | sequential | 2.58 / 2.58 | **0.71 / 0.75** | 3.8 / 3.8 | 1.2 / 1.3 |
| 4096x6256 @46 | random | 2.50 / 2.54 | **0.67 / 0.75** | 3.8 / 3.5 | 1.0 / 1.3 |

### Reads, native x86_64

| Cell | Scenario | Directory p50 | PMTiles p50 | Directory p99 | PMTiles p99 |
|---|---|---|---|---|---|
| 2048x2048 @256 | cold | 17.46 / 17.58 | 80.32 / 63.47 | 20.2 / 53.7 | 163.1 / 174.0 |
| 2048x2048 @256 | warm | 11.74 / 13.52 | 16.22 / 14.15 | 24.0 / 16.8 | 88.6 / 23.8 |
| 2048x2048 @256 | sequential | 11.71 / 12.87 | 17.11 / 15.11 | 66.5 / 23.3 | 28.5 / 26.7 |
| 2048x2048 @256 | random | 11.26 / 12.43 | 16.42 / 14.46 | 19.9 / 26.7 | 157.8 / 26.5 |
| 8192x8192 @256 | cold | 21.13 / 17.18 | 70.87 / 73.33 | 25.6 / 23.3 | 127.1 / 86.8 |
| 8192x8192 @256 | warm | 14.94 / 13.38 | **12.67 / 12.07** | 18.6 / 17.3 | 15.4 / 14.5 |
| 8192x8192 @256 | sequential | 17.43 / 17.21 | **16.56 / 10.53** | 21.9 / 20.6 | 31.9 / 18.2 |
| 8192x8192 @256 | random | 18.86 / 18.20 | **18.56 / 12.70** | 24.1 / 28.3 | 24.7 / 20.4 |
| 16384x16384 @256 | cold | 21.13 / 17.45 | 115.64 / 169.80 | 28.5 / 39.0 | 213.8 / 269.2 |
| 16384x16384 @256 | warm | 12.67 / 10.97 | **9.97 / 7.92** | 15.3 / 25.2 | 14.7 / 11.0 |
| 16384x16384 @256 | sequential | 15.25 / 15.20 | **11.75 / 11.60** | 23.7 / 22.2 | 19.4 / 17.6 |
| 16384x16384 @256 | random | 17.99 / 17.57 | **15.12 / 13.47** | 26.7 / 24.3 | 33.9 / 28.3 |
| 8192x8192 @64 | cold | 6.07 / 6.46 | 112.75 / 124.00 | 52.6 / 24.8 | 195.5 / 193.9 |
| 8192x8192 @64 | warm | 3.48 / 3.57 | **0.72 / 0.64** | 3.8 / 4.7 | 1.0 / 0.9 |
| 8192x8192 @64 | sequential | 5.38 / 5.39 | **1.34 / 1.26** | 6.4 / 6.4 | 3.4 / 3.1 |
| 8192x8192 @64 | random | 5.48 / 5.46 | **1.65 / 1.56** | 10.4 / 10.3 | 5.2 / 5.0 |
| 4096x6256 @46 | cold | 5.42 / 5.83 | 320.46 / 326.39 | 21.4 / 22.4 | 583.9 / 649.8 |
| 4096x6256 @46 | warm | 3.26 / 3.40 | **0.56 / 0.56** | 3.5 / 3.7 | 0.7 / 0.9 |
| 4096x6256 @46 | sequential | 4.57 / 4.64 | **1.09 / 1.09** | 5.4 / 5.5 | 3.2 / 3.3 |
| 4096x6256 @46 | random | 4.52 / 4.55 | **1.38 / 1.38** | 9.1 / 9.7 | 1.9 / 2.0 |

The bar #986 sections 9 and 18 set is competitive local read latency, not
beating a raw `pread` on every single local read, and the archive clears it on
both machines. Where it is furthest ahead is the cell with the most tiles, which
is the direction that matters: at 21851 tiles a warm lookup is 0.46 us against
1.54 on the M5 and 0.72 against 3.48 on the 8505, because one `pread` into a file
that is already open does not care how many tiles the pyramid has and a path
resolution through a directory tree does.

Cold is the row it loses, on both machines, and the next section is about why.

### The ramp, measured to its peak

A cold open costs what the archive's root directory costs to decode, and issue
#1021 fitted a line through three cells and put the peak at about 277 us without
ever measuring a root near the cutoff. The brink cell measures it. Its root holds
16369 entries, fourteen under the 16383 the writer still keeps flat, and opening
it costs **245.29 / 243.50 us** on the M5 and **320.46 / 326.39 us** on the 8505,
against 2.83 and 5.42 us for a tree the same size.

| Cell | Root entries | arm64 gap | arm64 ratio | x86_64 gap | x86_64 ratio |
|---|---|---|---|---|---|
| 2048x2048 @256 | 93 | 6.00 / 4.08 | 1.86x / 1.60x | 62.86 / 45.89 | 4.60x / 3.61x |
| 8192x8192 @256 | 1373 | 31.12 / 32.29 | 4.52x / 5.81x | 49.75 / 56.15 | 3.35x / 4.27x |
| 16384x16384 @256 | 5469 | 84.88 / 85.38 | 11.72x / 12.26x | 94.51 / 152.35 | 5.47x / 9.73x |
| 8192x8192 @64 | 6 | 66.75 / 65.58 | 22.65x / 21.71x | 106.68 / 117.53 | 18.57x / 19.18x |
| 4096x6256 @46 | 16369 | 242.46 / 240.67 | 86.55x / 85.92x | 315.04 / 320.56 | 59.18x / 56.01x |

The gap is the PMTiles cold p50 minus the directory cold p50, and it is not a
clean function of the root size, because it also carries one `pread` of a tile
whose size depends on the cell rather than on the root. The 93-entry cell hands
back 87 KB tiles and the 16369-entry cell hands back tiles of about 3 KB, so a
line fitted through the gap column is fitting two variables at once.
That is what the split is for.

### The cold open, phase by phase

PMTiles p50 per phase, microseconds. `combined` is the single `read_cold` row and
`sum` adds the six phase medians, so the two columns reconciling is what says the
split measures the same work. `sum` is not quite the statistic the guard uses:
that one takes the median of the per-iteration sums, which is the stricter
reading and comes out a few tenths of a percent different here.

**Native arm64**

| Cell | Root entries | open | header | root fetch | inflate | decode | lookup | sum | combined |
|---|---|---|---|---|---|---|---|---|---|
| 2048x2048 @256 | 93 | 0.67 / 0.67 | 0.25 / 0.25 | 0.21 / 0.21 | 2.96 / 2.96 | 1.25 / 1.33 | 4.46 / 4.12 | 9.79 / 9.54 | 12.96 / 10.92 |
| 8192x8192 @256 | 1373 | 0.71 / 0.71 | 0.25 / 0.25 | 0.25 / 0.25 | 6.25 / 6.21 | 18.29 / 17.75 | 6.96 / 7.54 | 32.71 / 32.71 | 39.96 / 39.00 |
| 16384x16384 @256 | 5469 | 0.71 / 0.71 | 0.25 / 0.25 | 0.29 / 0.29 | 12.62 / 12.67 | 68.71 / 69.54 | 6.92 / 7.12 | 89.50 / 90.59 | 92.79 / 92.96 |
| 8192x8192 @64 | 6 | 0.71 / 0.71 | 0.25 / 0.25 | 0.21 / 0.21 | 2.12 / 2.08 | 0.12 / 0.12 | 67.33 / 65.29 | 70.75 / 68.67 | 69.83 / 68.75 |
| 4096x6256 @46 | 16369 | 0.71 / 0.75 | 0.25 / 0.25 | 0.50 / 0.50 | 32.08 / 32.00 | 209.12 / 205.83 | 0.88 / 0.96 | 243.54 / 240.29 | 245.29 / 243.50 |

**Native x86_64**

| Cell | Root entries | open | header | root fetch | inflate | decode | lookup | sum | combined |
|---|---|---|---|---|---|---|---|---|---|
| 2048x2048 @256 | 93 | 4.52 / 3.67 | 0.99 / 0.91 | 0.91 / 0.75 | 16.61 / 13.68 | 3.61 / 3.17 | 27.38 / 24.04 | 54.03 / 46.23 | 80.32 / 63.47 |
| 8192x8192 @256 | 1373 | 2.11 / 2.25 | 0.45 / 0.52 | 0.42 / 0.44 | 11.50 / 12.10 | 20.32 / 21.65 | 20.53 / 20.81 | 55.33 / 57.76 | 70.87 / 73.33 |
| 16384x16384 @256 | 5469 | 2.04 / 2.11 | 0.44 / 0.49 | 0.49 / 0.48 | 21.27 / 20.51 | 82.08 / 80.26 | 17.46 / 16.91 | 123.78 / 120.77 | 115.64 / 169.80 |
| 8192x8192 @64 | 6 | 1.84 / 1.82 | 0.41 / 0.41 | 0.38 / 0.38 | 5.77 / 5.79 | 0.17 / 0.17 | 81.17 / 81.27 | 89.75 / 89.82 | 112.75 / 124.00 |
| 4096x6256 @46 | 16369 | 2.22 / 2.17 | 0.54 / 0.45 | 0.94 / 0.84 | 69.04 / 70.10 | 230.43 / 230.32 | 1.31 / 1.08 | 304.49 / 304.95 | 320.46 / 326.39 |

Three things fall out of those two tables and none of them is visible in the
combined row.

**The decode is the ramp, and its slope is the same on both machines.** Fit the
decode column against the root size and it lands on a straight line, at
**12.65 ns an entry** on arm64 and **13.95 ns an entry** on x86_64:

```text
arm64    decode_us = 0.25 + 12.65 ns/entry * root
     93 entries: measured   1.29   predicted   1.43
   1373 entries: measured  18.02   predicted  17.63
   5469 entries: measured  69.13   predicted  69.46
  16369 entries: measured 207.48   predicted 207.40

x86_64   decode_us = 2.73 + 13.95 ns/entry * root
     93 entries: measured   3.39   predicted   4.02
   1373 entries: measured  20.99   predicted  21.88
   5469 entries: measured  81.17   predicted  79.00
  16369 entries: measured 230.38   predicted 231.02
```

Two boxes whose other read rows sit anywhere from 1.5x to 3x apart agree on this
slope to within 10%, and the brink point sits on the line rather than being
extrapolated to. That is the number an optimisation has to move, and it is
close to the 11.6 to 12.9 ns an entry the micro measurement further down this
document already recorded for `deserialize_entries`.

**The inflate is not free and it is not the ramp either.** It is 32 us on arm64
and 69 on x86_64 at the brink, 13% and 22% of the whole open, and it
grows with the compressed size rather than with the entry count. Anyone going at
the decode should know the gzip is the next thing behind it.

**The open, the header and the ranged read are nothing.** On arm64 they are 0.71,
0.25 and 0.50 us at the brink, about 0.6% of the open between them. A fix that
batched or elided them would be fixing the cheap part.

The phases reconcile with the combined row within 1% on the cells whose root
dominates (the guard's own reading is -0.7% and -0.9% at the brink on arm64,
-4.3% on x86_64) and drift on the cells where it does not: -24% and -13% on the
93-entry cell, where the whole open is 11 us and a few hundred nanoseconds of
per-iteration overhead is a fifth of it.
`the_cold_split_accounts_for_the_whole_combined_row` used to allow 25% and pick
a cell with about 1400 entries for exactly that reason, which is a tolerance
chosen to fit the instrument rather than the claim. It no longer compares
durations at all. A 25% allowance against a spread that reached 39 points
between consecutive runs at one commit on one machine went red under load and
green when idle, for reasons that had nothing to do with the code, and widening
it would only move the load at which it lies. The guard now reconciles the split
against a real cold open by the byte ranges each one reads and the values each
one decodes, which is the same claim without the instrument.

The percentages above stay because they are worth knowing, not because anything
asserts them. The quantitative form of the claim, that the six durations sum to
the combined duration, wants repetitions and a dispersion the run itself
measured, and belongs with the storage family in `libviprs-bench` rather than in
a guard that has to pass on every CI job.

### The thread ladder, and where it knees

`read_concurrent` at 1, 2, 4 and 8 threads, one row each, with T=1 as the
control. All four points run on the same reader `read_sequential` and
`read_random` have already walked, in that order, so every point is a warm
reader and each inherits whatever the previous left behind.

**Native arm64**, PMTiles p99, microseconds:

| Cell | T=1 | T=2 | T=4 | T=8 |
|---|---|---|---|---|
| 2048x2048 @256 | 29.2 / 36.5 | 34.8 / 49.2 | 48.8 / 31.8 | 28.0 / 33.6 |
| 8192x8192 @256 | 15.6 / 10.7 | 18.4 / 13.0 | 21.0 / 21.1 | 25.7 / 27.7 |
| 16384x16384 @256 | 12.9 / 15.9 | 17.9 / 17.1 | 32.5 / 26.8 | 39.7 / 77.2 |
| 8192x8192 @64 | 1.8 / 1.9 | 2.0 / 2.0 | **8.8 / 8.2** | **38.5 / 38.0** |
| 4096x6256 @46 | 1.2 / 1.5 | 1.3 / 1.6 | 1.3 / 2.1 | 2.1 / 2.3 |

**Native x86_64**, PMTiles p99, microseconds:

| Cell | T=1 | T=2 | T=4 | T=8 |
|---|---|---|---|---|
| 2048x2048 @256 | 89.4 / 85.4 | 110.7 / 155.6 | 99.9 / 86.0 | 119.2 / 106.5 |
| 8192x8192 @256 | 30.5 / 19.2 | 38.5 / 32.4 | 43.8 / 58.9 | 109.5 / 62.0 |
| 16384x16384 @256 | 25.8 / 31.2 | 23.9 / 26.9 | 39.6 / 47.4 | 78.3 / 56.2 |
| 8192x8192 @64 | 2.9 / 2.0 | 3.1 / 3.3 | 4.0 / 5.6 | **13.1 / 21.8** |
| 4096x6256 @46 | 2.3 / 2.2 | 2.5 / 2.4 | 4.2 / 2.8 | 5.2 / 3.1 |

Every cell's p99 rises with the thread count, because eight threads competing for
cache and for the scheduler cost something whatever the storage is. One cell
rises differently. 8192x8192 @64 goes from 1.8 to 38.5 us on arm64, a factor of
21, where the next worst cell moves by 3x and the brink cell by 1.8x. It is also
the only cell whose archive has leaf directories, and the brink cell has the
biggest root in the sweep and barely moves, which rules the root out: the cost is
the single `Mutex` around the leaf cache that every lookup on a leaf-bearing
archive takes.

The curve knees in a different place on the two machines. On arm64 it is at
**four threads**, where p99 goes 2.0 to 8.8 and then to 38.5; on x86_64 the first
three points are flat and the move is all at **eight**, 5.6 to 13.1 and 21.8. The
box with fewer cores knees later, which is the wrong way round for
oversubscription and the right way round for a lock: the M5's threads get back to
the mutex sooner, so four of them queue where six slower ones do not.

Eight threads run even on a box with fewer cores. That is oversubscription rather
than parallelism and it is left in deliberately, because the envelope's
provenance records `ncpu` and a reader can see which points had a core to
themselves.

### Which cells have leaf directories

Measured rather than assumed, by asking each archive's root what it holds. Same
answer on both machines, because it is a property of the writer:

| Cell | Tiles | Root holds |
|---|---|---|
| 2048x2048 @256 | 93 | 93 tile entries, no leaves |
| 8192x8192 @256 | 1373 | 1373 tile entries, no leaves |
| 16384x16384 @256 | 5469 | 5469 tile entries, no leaves |
| 8192x8192 @64 | 21851 | 6 pointers, all of them leaves |
| 4096x6256 @46 | 16369 | 16369 tile entries, no leaves, 14 under the cutoff |

Under 16384 entries the writer keeps the whole directory in the root, so only the
fourth cell exercises the leaf lookup, the leaf cache and the second ranged read
at all. A sweep without it measures one half of the read path and reports it as
the read path, which is exactly what the first version of this one did.
`the_large_profile_reaches_the_leaf_directory_path` asserts the sweep still
crosses the cutoff, so an edit to the cell list cannot quietly drop it.

### What the caveats mean for the new rows

Nothing above drops the caveats the older rows carried, and three of them now
reach further than they did.

**Cold still means a cold reader and not a cold page cache.** Dropping the OS page
cache needs root on Linux and has no portable equivalent, so the harness does not
claim to have done it, and that is true of the six phase rows as well: every one
of them reads a file some earlier iteration has already read. The gap between the
cold and the warm rows is the open cost plus the in-process caches, which is the
part libviprs controls.

**`read_random` still walks a reader `read_sequential` has just warmed,** and now
so does every point on the thread ladder, because all four run on that same
reader afterwards. None of those rows is a cold-cache measurement and none of
them is comparable to one.

**`peak_rss_mb` is still `/proc`-only** and publishes `null` where the platform
has no answer. Both machines above are Linux containers so every row has one. All
six phase rows carry the same number, sampled once after the split pass finished,
because a phase does not get a process of its own and a high-water mark cannot be
attributed to a part of one.


### The tables from here down predate the provenance block

Everything below this line was measured before the harness recorded a host, so
none of it says which architecture it ran on and I am not going to invent one.
They are all Docker Desktop containers on the same Apple Silicon Mac, and whether
a given one was the arm64 or the emulated amd64 image is exactly the thing that
was never written down. They are kept because what they show is a ratio or a
shape rather than a wall-clock figure, and those survive the uncertainty. Anything
re-measured from now on carries its provenance in the export.

### What an unhelpful filesystem does to this

The same cheap profile, run earlier with the scratch directory on a bind mount
from the host (Docker Desktop's virtiofs) instead of the container's own
filesystem, and on a busy machine:

| Scenario | Directory p50 | PMTiles p50 |
|---|---|---|
| Warm | 555.1 | 17.0 |
| Sequential | 868.0 | 15.9 |
| Random | 619.0 | 18.0 |
| Concurrent | 15340.0 | 21.5 |

The archive barely moves, because a lookup is a `pread` into one open file
whatever the filesystem is. The tree collapses, because every lookup is a fresh
path resolution and that is the operation a virtualised or networked filesystem
is worst at. Neither set is the "true" one. They are the two ends of the range
the storage decision sits in, and the second is the one that looks like an
object store.

### The optimisation pass, and what the measurement asked for

Issue #993 asks for one **as guided by the measurements**, over the finalize
merge buffering, the reader's directory page caching and the IO buffering. Two
of those three the measurements had nothing to say about:

- the external merge already reads every run through one file descriptor with a
  capped fan-in, and the bounded-memory tests measure it at 650 KiB of live heap
  on 262144 tiles;
- the payload copy already runs through a 64 KiB buffer, and the payloads
  themselves arrive in `write_all` calls larger than any buffer would hold.

The third was a real finding, and only the 64 pixel tile cell could see it. On
that archive, random access was **thirteen to fifteen times slower than
sequential access over the same 20000 coordinates**, and slower than the
directory backend:

| 20000 lookups, 8192x8192 @64 | Before | After |
|---|---|---|
| PMTiles random, wall | 1699.9 / 1013.3 ms | 55.0 / 38.9 ms |
| PMTiles sequential, wall | 127.3 / 65.9 ms | 38.0 / 43.1 ms |
| Random over sequential | 13.4x / 15.4x | 1.45x / 0.90x |
| PMTiles random, p99 | 1486.2 / 608.6 us | 5.6 / 5.6 us |
| Directory random, wall (control) | 204.2 / 345.2 ms | 135.5 / 126.6 ms |

The "after" column was taken at a cache of sixteen, which is what this fix
first shipped as. The section below is why it is sixty-four now, and on this
archive, which has six leaves, both sizes hold every leaf and the numbers do
not move between them.

Two runs of each, and the two numbers in every cell are those two runs. That
whole table was taken on a busy host, which is why the ratio row is the one to
read: the machine got less busy between the two sets and moved the directory
control with it, while random over sequential is measured inside one process
seconds apart so the contention cancels out of it. On the idle machine the
sweep it was compared against ran on, the same ratio was 24.90 over 21.25, or
1.17x. The quiet runs at the top of this document put it at 1.19x and 1.08x on
arm64 and 1.23x and 1.24x on x86_64, so the fix is still holding.

The cause was the reader's leaf cache holding four decoded leaves. That is the
right size for the clustered walk it was written for, and the wrong size for
random access: this archive has six leaves, an LRU of four over six uniformly
random leaves misses about a third of the time, and every miss pays a ranged
read **and** a decode of a 4096-entry directory.

### The size is a step, not a slope

The first fix here raised the count to sixteen and said that past sixteen
leaves the cache "degrades the way any LRU does rather than falling off a
cliff". That is false, and measuring it is what says so. The miss rate of an
LRU of `k` over `N` uniformly random leaves is exactly `1 - k/N`, because the
cache holds the `k` most recently referenced distinct leaves and every leaf is
equally likely to be one of them. On fabricated archives of 4096-entry leaves,
20000 random lookups each:

| leaves | tiles | cache of 16 | cache of 64 |
|---|---|---|---|
| 16 | 65536 | 0.38 us, 0.1% miss | 0.36 us |
| **17** | **69632** | **7.37 us, 5.9% miss** | 0.37 us |
| 24 | 98304 | 37.61 us, 32.9% miss | 0.43 us |
| 64 | 262144 | 93.40 us, 75.1% miss | 0.69 us |
| 256 | 1048576 | 112.12 us, 93.6% miss | 70.75 us |

Sixteen leaves to seventeen is a nineteenfold jump for one more leaf. The cliff
does not soften with size, it moves.

So the count is derived from the memory bound rather than picked. It is
`MAX_CACHED_LEAF_ENTRIES / 4096`, which is **64**, and a `const _` assertion in
`src/pmtiles/reader.rs` holds the two together so they cannot disagree again.
They did disagree: sixteen leaves at 4096 entries is 65536, a quarter of the
262144-entry budget, so the count bound always bit first and the budget never
bound at all on any archive this crate writes. Going to 64 costs nothing the
budget had not already declared acceptable.

`MAX_CACHED_LEAF_ENTRIES` is the memory bound, at 262144 entries or about 6
MiB, and that figure is now the real ceiling. It leads with the budget because a
count of leaves is not a bound at all: one leaf may decode to as many entries as
`MAX_DIRECTORY_BYTES` allows, which is about a million. A leaf over the whole
budget by itself is handed back to the lookup and not cached, where it used to
be kept on the argument that the lookup holds it anyway. That is true of the
`Arc` and not of the cache slot, and the difference between the two is a 6 MiB
ceiling and a 24 MiB one.

### What a miss actually costs

Not the gzip, which is what this document, the CHANGELOG and the reader's own
rustdoc all said. Measured on a realistic leaf, 9157 stored bytes inflating to
22647: `deserialize_entries` is 52 to 68 us, the gzip inflate is 32 to 35, and
the whole miss is 84 to 103. The varint decode is about 62% of it. Per-entry
decode cost is flat at 11.6 to 12.9 ns from 64 entries to 16384, so this is the
varint loop itself at roughly 3 ns a varint rather than cache locality across
the column passes. Anyone who wants a cheaper miss should go at the decode.

### The guards

`every_leaf_of_a_multi_leaf_archive_stays_cached` in
`tests/pmtiles_index_only_reads.rs` **counts reads rather than timing them**: an
eight-leaf fabricated archive, walked once to warm and once backwards to check,
has to answer the second pass with one read per tile and no directory reads at
all. A timing assertion in that position would have been a benchmark pretending
to be a guard, and it would say something different on every machine.

`the_miss_rate_over_a_cache_too_small_tracks_the_cache_size` is the other side,
and it is the side that was unreachable before: the guard above used to
compile-assert that its own archive fitted the cache, so the only interesting
case, more leaves than the cache holds, could not be written. It builds a
96-leaf archive, walks 4096 seeded random lookups and asserts the miss rate
tracks `1 - k/N`: measured 0.3323 against a predicted 0.3333. It also pins the
cache size against the writer's own leaf size, so somebody re-hardcoding a count
fails there rather than shrinking a test archive and staying green.

## The bounded-memory proof

`cargo test --test pmtiles_bounded_memory` installs a counting global allocator
and measures live heap bytes across a whole write. Live heap is the only basis
on which "bounded" has a truth value: the engine's `MemoryTracker` charges
raster buffers and knows nothing about the writer's own vectors, and process
RSS is a high-water mark that cannot show a peak falling.

Three assertions, because one number proves nothing:

- four times the tiles at a fixed sort buffer must not move the peak
- sixty-four times the sort buffer at a fixed tile count must move the peak by
  roughly the buffer's own size, which is the control that stops the first
  assertion passing on a writer whose peak is a constant
- the absolute peak must sit under a formula built from the sort-buffer size,
  the distinct-payload count and a fixed overhead

Each also asserts the header the writer produced, because a run that dropped
its tiles on the floor would beat all three bounds.

The allocator's `realloc` charges the new block before discharging the old one,
so a growing `Vec` is counted twice for an instant and a peak dominated by one
of them reads high. That is deliberate, because conservative is the right
direction for an upper-bound test, and it is worth knowing when reading a
number: on the 262144-record sort buffer this order reports 9454340 bytes and
discharging first reports 6809585. The 4096-record cell is identical either way
and the absolute-bound cell moves by 1550 bytes on 650215, which is why the one
figure this document publishes does not move.

The bound is **not** independent of everything. `src/pmtiles/writer.rs` says so
itself: the content-hash table, the payload table and the final-offset lookup
all scale with the number of *distinct payloads*, and no amount of spilling
changes that. A synthetic pyramid of identical tiles has one distinct payload
however many tiles it has, which is exactly how a benchmark hides this, so the
tests cycle over several payloads rather than repeating one.

### The offsets past 4 GiB

Entry offsets are `u64`. An implementation that narrowed one to `u32` anywhere
would be invisible until an archive crossed 4 GiB, so both halves are covered.

The read half is free and runs on every CI job:
`cargo test --test pmtiles_index_only_reads` fabricates the header and the
directories of a 6 GiB archive through the crate's own `Header::encode` and
`serialize_entries`, serves them through a `RangeReader` that synthesises tile
bytes and counts every request, and asserts that a request really landed past
`u32::MAX`. The same source is what proves reads are index-only: opening the
archive fetches under 16 KiB, a root-addressed tile is exactly one read of its
own length, a leaf-addressed tile is the leaf and the tile with the leaf cached
after that, and a whole workload never touches the metadata section.

The write half really has to stage 4 GiB, so it is opt-in:

```sh
cargo test --release --test pmtiles_bounded_memory -- --ignored --nocapture
```

That profile writes 272 distinct 16 MiB payloads through a sink that keeps the
position and drops the bytes, so the writer performs every copy, seek and
offset computation while the measurement stays a measurement of the writer
rather than of the filesystem. It asserts `tile_data_length` is past `u32::MAX`
and that the peak is still under the same formula, with nothing added for the
harness: the payload buffer is allocated before the baseline is taken, and
there is a second assertion that the peak is under one payload so the bound
cannot quietly start paying for it again. The run:

```text
4 GiB profile: tile_data_length=4563402752 peak_heap_bytes=410369 bound=4425728
```

## Release readiness

`cargo test --test pmtiles_release_readiness` keeps the four claims honest:

1. **PMTiles is in every build.** The `pmtiles`, `sink_pmtiles` and
   `pyramid_reader` modules are not behind a Cargo feature, so "PMTiles is the
   default storage" cannot become false by one `cfg` line.
2. **The release notes describe what shipped.** CHANGELOG.md and MIGRATION.md
   both name `PyramidStorage`, and MIGRATION.md states what `default()` answers
   and which variant keeps the old behaviour.
3. **A publish can be rehearsed.** `publish.yml` offers a `dry_run` input, a
   step honours it with `cargo publish --dry-run --locked`, and the real upload
   stays behind the published-contract gate.
4. **This document is runnable.** Every `--test` command above names a test
   file that exists, and the column table above is checked against the field
   list the harness actually emits.

The feature matrix, the doctests and the intra-doc link gate are not repeated
here. They are `tests/ci_feature_coverage.rs`, `make doc` and
`tests/doc_link_gate.rs`, and `make ci` runs the real job list rather than a
copy of it.
