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
across every available core.

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

### Root-only archives, and the one cell that is not

Under 16384 directory entries the writer puts the whole directory in the root,
so an archive of a few thousand tiles never exercises the leaf lookup, the leaf
cache or the second ranged read. Three of the four cells in the large profile
are in that regime and so is the CI cell.

The fourth is 8192 pixels at a **64 pixel** tile: 21845 tiles, past the cutoff,
so its archive really has leaf directories. Reaching that through the tile size
rather than through a bigger canvas is deliberate, because 21845 tiles at 256
pixel tiles needs a 32768 pixel source and that is a 3.2 GB raster, while what
the read path cares about is the directory shape rather than the pixels behind
it. `the_large_profile_reaches_the_leaf_directory_path` asserts the sweep still
crosses the cutoff, so a future edit to the cell list cannot quietly drop the
only cell that covers half the read path.

## Running it

The cheap profile is the default and takes seconds:

```sh
cargo test --release --test pmtiles_benchmarks -- --ignored --nocapture
```

The large profile walks four cells, up to 16384x16384 pixels and up to 21845
tiles, and takes minutes. It is opt-in on purpose, since a benchmark nobody runs
because it is too expensive is a benchmark nobody runs:

```sh
LIBVIPRS_BENCH_PROFILE=large \
LIBVIPRS_BENCH_JSON=/tmp/pmtiles-benchmarks.json \
cargo test --release --test pmtiles_benchmarks -- --ignored --nocapture
```

`LIBVIPRS_BENCH_PROFILE` selects `ci` (the default) or `large`.
`LIBVIPRS_BENCH_JSON` chooses where the export lands; without it the export
goes to `target/pmtiles-benchmarks.json`.

Run it in the Linux container rather than on a developer machine if the peak
RSS column matters. It comes from `/proc/self/status`, which exists on Linux
and nowhere else this crate builds for, and a platform with no answer reports
`0.0` rather than a number invented to fill the column.

### Every measurement is a fresh process

Peak RSS is a high-water mark the kernel never lowers on its own, so two
backends measured in one process hand the second one the first one's peak. The
benchmark test re-executes its own test binary once per cell and each cell
reports its own process's numbers; inside a cell, the read phase resets the
high-water mark through `/proc/self/clear_refs` so generation's peak does not
become the read scenario's floor.

## The exported JSON

A top-level array of objects, one per measured row, written to the path
`LIBVIPRS_BENCH_JSON` names. The first twelve fields are spelled exactly as
`scalability_results.json` spells them, so the libviprs.org renderer and the
`ScalabilityPoint` deserialiser in libviprs-bench read this file with no second
code path. The rest are additive: an unknown key is ignored by `serde_json` and
by the site's JavaScript.

| Column | Meaning |
|---|---|
| `width` | Source canvas width in pixels |
| `height` | Source canvas height in pixels |
| `megapixels` | `width * height / 1e6` |
| `tile_size` | Tile edge in pixels. Two rows at one canvas and two tile sizes are not the same measurement: the tile count, and so the archive's directory shape, follows from it |
| `engine` | `"pmtiles"` or `"directory"`. Named `engine` because that is the key the renderer groups on |
| `concurrency` | Threads the row was measured at. 1 everywhere except `read_concurrent` |
| `wall_time_ms` | Wall-clock milliseconds for the whole row |
| `tracked_memory_mb` | The engine's own `MemoryTracker` peak: raster buffers, nothing else. 0 for a read row |
| `peak_rss_mb` | Process peak resident set for this phase. 0 where the platform has no answer |
| `tiles_produced` | Tiles written, or tiles read back for a read row |
| `tiles_per_second` | `tiles_produced` over wall time |
| `tiles_per_second_per_mb` | Throughput per peak-RSS megabyte. Higher is better |
| `resource_cost` | RSS-megabyte-seconds per tile. Lower is better |
| `scenario` | `generate`, `read_cold`, `read_warm`, `read_sequential`, `read_random` or `read_concurrent` |
| `storage` | The same value as `engine`, under the name that says what it is |
| `profile` | `ci` or `large` |
| `output_bytes` | Bytes the pyramid occupies on disk |
| `filesystem_entries` | Filesystem entries it occupies, directories included. 1 for an archive |
| `bytes_fetched` | Tile payload bytes the row's lookups returned. 0 for a generation row |
| `p50_latency_us` | Median per-lookup latency in microseconds. 0 for a generation row |
| `p99_latency_us` | 99th-percentile per-lookup latency in microseconds. 0 for a generation row |

The two ratio columns are derived exactly the way the existing scalability
producer derives them, both on the peak-RSS basis, so a row here means the same
thing as a row there. Both are 0 when the RSS basis is unavailable, which is
the same answer that producer gives when its denominator is zero.

## The numbers, as measured

Run on 2026-09-12 in the Linux container (`rust:1.98.1`, 8 CPUs, 15.6 GB),
release profile, cheap profile, on the container's own filesystem. A gradient
source at 2048x2048 with 256 pixel tiles, which the planner turns into 93 tiles
over twelve levels.

**The host was not quiet.** A second Docker gate was running throughout, at a
load average of about 9.5 on 8 CPUs. Every absolute number below is therefore
slower than the machine can go, and the concurrent row is the one to trust
least, since it is the only scenario competing for cores with something else.
The comparison survives it better than the absolutes do, because the two
backends are measured one after the other inside one sweep under the same
contention, but a figure quoted anywhere that matters should be re-measured on
an idle machine. Re-running is one command and it is the one at the top of this
page.

### Generation

| Measurement | Directory | PMTiles |
|---|---|---|
| Wall time | 357.8 ms | 434.4 ms |
| Throughput | 259.9 tiles/s | 214.1 tiles/s |
| Engine working set | 12.0 MB | 12.0 MB |
| Process peak RSS | 34.0 MB | 34.1 MB |
| Output bytes | 8 147 032 | 8 147 559 |
| Filesystem entries | 129 | **1** |

PMTiles generation costs 21% more wall time, and that is structural rather than
an oversight. The root directory has to fit the spec's first 16 KiB, and an
entry's offset is not known until the tiles are sorted, so the payloads are
staged and then copied into the archive: roughly twice the archive's size in
writes for the same tiles. The archive is 527 bytes larger than the sum of the
tree's files, which is the header, the directory and the metadata.

### Reads

93 lookups per scenario (64 for the cold and warm rows), microseconds per
lookup:

| Scenario | Directory p50 | PMTiles p50 | Directory p99 | PMTiles p99 |
|---|---|---|---|---|
| Cold | 14.9 | 27.9 | 3407.7 | 5687.5 |
| Warm | 10.7 | 12.5 | 174.3 | 403.6 |
| Sequential | 10.2 | 13.5 | 2379.0 | 515.3 |
| Random | 12.3 | 11.7 | 4338.9 | **63.8** |
| Concurrent | 14.5 | 11.7 | 8225.2 | 1817.4 |

Whole-scenario wall time, milliseconds: sequential 6.11 against 2.11, random
15.81 against 1.20, concurrent 65.77 against 37.75, all three in the archive's
favour.

The median is where the epic set the bar (#986 sections 9 and 18: competitive
local read latency, not necessarily beating a raw `pread` on every single
local read), and the medians are within a few microseconds of each other in
both directions. The tail is where the archive wins outright: random access at
a p99 of 63.8 microseconds against 4338.9 is a 68x difference, and it is the
difference between one `pread` into a file that is already open and a path
resolution through a directory tree the kernel has to walk.

Cold is the one row the archive loses, by 13 microseconds, and that is its open
cost: the header fetch and the root-directory fetch a client pays once before
its first tile. A tree pays nothing to open because there is nothing to read.

### What an unhelpful filesystem does to this

The same cheap profile, run with the scratch directory on a bind mount from the
host (Docker Desktop's virtiofs), rather than on the container's own
filesystem:

| Scenario | Directory p50 | PMTiles p50 |
|---|---|---|
| Warm | 555.1 | 17.0 |
| Sequential | 868.0 | 15.9 |
| Random | 619.0 | 18.0 |
| Concurrent | 15340.0 | 21.5 |

The archive barely moves, because a lookup is a `pread` into one open file
whatever the filesystem is. The tree collapses, because every lookup is a fresh
path resolution and that is the operation a virtualised or networked filesystem
is worst at. Neither set of numbers is the "true" one. They are the two ends of
the range the storage decision sits in, and the second is the one that looks
like an object store.

### The large profile

Four cells, same container, same contention. Generation, with the number the
whole epic is about in the last column:

| Cell | Backend | Wall time | Tiles/s | Output bytes | Filesystem entries |
|---|---|---|---|---|---|
| 8192x8192 @256, 5467 tiles | directory | 5487.7 ms | 250.2 | 130 159 233 | 1459 |
| 8192x8192 @256 | pmtiles | 4280.7 ms | 320.7 | 130 160 332 | **1** |
| 16384x16384 @256, 21851 tiles | directory | 18846.6 ms | 290.2 | 520 598 291 | 5620 |
| 16384x16384 @256 | pmtiles | 25081.8 ms | 218.0 | 520 600 896 | **1** |
| 8192x8192 @64, 21851 tiles | directory | 11088.3 ms | 1970.6 | 133 748 027 | 22127 |
| 8192x8192 @64 | pmtiles | 7278.7 ms | 3002.0 | 133 758 824 | **1** |

Generation goes both ways and the spread between repeats of one cell is wider
than the gap between the two backends, which is what a loaded host looks like.
The entry counts are exact and do not move: 22127 filesystem entries against 1,
for the same 133 MB of tiles.

### The optimisation pass, and what the measurement asked for

Issue #993 asks for an optimisation pass **as guided by the measurements**, over
the finalize merge buffering, the reader's directory page caching and the IO
buffering. Two of those three the measurements had nothing to say about:

- the external merge already reads every run through one file descriptor with a
  capped fan-in, and the bounded-memory tests measure it at 650 KiB of live heap
  on 262144 tiles;
- the payload copy already runs through a 64 KiB buffer, and the payloads
  themselves arrive in `write_all` calls larger than any buffer would hold.

The third was a real finding, and only the 64 pixel tile cell could see it,
because it is the only cell whose archive has leaf directories at all. On that
archive, random access was **thirteen to fifteen times slower than sequential
access over the same 20000 coordinates**, and slower than the directory
backend:

| 20000 lookups, 8192x8192 @64 | Before | After |
|---|---|---|
| PMTiles random, wall | 1699.9 / 1013.3 ms | 55.0 / 38.9 ms |
| PMTiles sequential, wall | 127.3 / 65.9 ms | 38.0 / 43.1 ms |
| Random over sequential | 13.4x / 15.4x | 1.45x / 0.90x |
| PMTiles random, p99 | 1486.2 / 608.6 us | 5.6 / 5.6 us |
| Directory random, wall (control) | 204.2 / 345.2 ms | 135.5 / 126.6 ms |

Two runs of each, and the two numbers in every cell are those two runs.

The ratio row is the one to read. The host was busy and got less busy between
the two sets, so the absolutes moved for the directory control too, and an
improvement claimed off the absolutes alone would be partly a claim about the
machine. Random over sequential is measured inside one process seconds apart,
so contention cancels: 13.4x and 15.4x before, 1.45x and 0.90x after.

The cause was the reader's leaf cache holding four decoded leaves. That is the
right size for the clustered walk it was written for, and the wrong size for
random access: this archive has six leaves, an LRU of four over six uniformly
random leaves misses about a third of the time, and every miss pays a ranged
read **and** a gzip inflate of a 4096-entry directory. The cache holds sixteen
now, which covers every leaf of an archive up to about 65000 tiles, and past
that it degrades the way an LRU does rather than falling off a cliff.

Raising a count is not free, so a count is no longer the only bound.
`LEAF_CACHE_ENTRY_BUDGET` caps the decoded entries the cache holds across every
leaf, because one leaf can decode to as many entries as `MAX_DIRECTORY_BYTES`
allows and sixteen of those would be hundreds of megabytes held by a reader
that was asked for a tile. The budget is 262144 entries, about 6 MiB, which an
ordinary archive never comes near.

`every_leaf_of_a_multi_leaf_archive_stays_cached` in
`tests/pmtiles_index_only_reads.rs` is the regression guard, and it **counts
reads rather than timing them**: an eight-leaf fabricated archive, walked once
to warm and once backwards to check, has to answer the second pass with one
read per tile and no directory reads at all. At a cache of four it answers with
twelve reads instead of eight, which is how that test was checked. A timing
assertion in that position would have been a benchmark pretending to be a
guard, and it would say something different on every machine.

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
and that the peak is still under the same formula.

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
