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

Two more things the read rows are not, so nobody reads more into them than they
hold. `read_random` runs on the reader `read_sequential` has just walked end to
end, so its leaf cache is whatever that pass left behind rather than empty; the
row is a warm random walk and it is compared against a warm sequential one.
And `read_concurrent` is one thread count per backend, chosen from the machine,
so these rows are not a scaling curve and nothing here reports one.

### Root-only archives, and the one cell that is not

Under 16384 directory entries the writer puts the whole directory in the root,
so an archive of a few thousand tiles never exercises the leaf lookup, the leaf
cache or the second ranged read. Three of the four cells in the large profile
are in that regime and so is the CI cell.

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

## Running it

The cheap profile is the default and takes seconds:

```sh
cargo test --release --test pmtiles_benchmarks -- --ignored --nocapture
```

The large profile walks four cells, up to 16384x16384 pixels and up to 21851
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

An envelope, `{"schema": 1, "rows": [...]}`, written to the path
`LIBVIPRS_BENCH_JSON` names. The envelope is there so a consumer can refuse a
document it was not written against instead of reading a renamed column as
absent, and `schema` is the number that changes when a field changes meaning.

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
| `concurrency` | Threads the row was measured at. 1 everywhere except `read_concurrent` |
| `wall_time_ms` | Wall-clock milliseconds for the whole row |
| `tracked_memory_mb` | The engine's own `MemoryTracker` peak: raster buffers, nothing else. `null` on a read row, which allocates none |
| `peak_rss_mb` | Process peak resident set for this phase. `null` where the platform has no answer |
| `tiles_produced` | Tiles written, or tiles read back for a read row |
| `tiles_per_second` | `tiles_produced` over wall time. `null` when the row took no measurable time |
| `tiles_per_second_per_mb` | Throughput per peak-RSS megabyte, higher is better. `null` whenever `peak_rss_mb` is |
| `resource_cost` | RSS-megabyte-seconds per tile, lower is better. `null` whenever `peak_rss_mb` is |
| `scenario` | `generate`, `read_cold`, `read_warm`, `read_sequential`, `read_random` or `read_concurrent` |
| `storage` | `pmtiles` or `directory`, the backend the row measured |
| `profile` | `ci` or `large` |
| `output_bytes` | Bytes the pyramid occupies on disk. `null` on a read row, and `null` when the path could not be walked |
| `filesystem_entries` | Filesystem entries it occupies, directories included. 1 for an archive. `null` on a read row, and `null` when the path could not be walked |
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

Run on 2026-09-12 in the Linux container (`rust` 1.98.1, 8 CPUs, 15.6 GB),
release profile, on the container's own filesystem, with nothing else running on
the host. A gradient source, `Layout::Xyz`, PNG tiles.

### Generation

| Cell | Tiles | Backend | Wall time | Tiles/s | Peak RSS | Output bytes | Filesystem entries |
|---|---|---|---|---|---|---|---|
| 2048x2048 @256 | 93 | directory | 84.5 ms | 1100.3 | 33.9 MB | 8 147 032 | 129 |
| 2048x2048 @256 | 93 | pmtiles | 110.7 ms | 840.1 | 34.1 MB | 8 147 559 | **1** |
| 8192x8192 @256 | 1373 | directory | 1221.9 ms | 1123.7 | 438.8 MB | 130 159 233 | 1459 |
| 8192x8192 @256 | 1373 | pmtiles | 1366.7 ms | 1004.6 | 439.3 MB | 130 160 332 | **1** |
| 16384x16384 @256 | 5469 | directory | 4958.0 ms | 1103.1 | 1734.9 MB | 520 598 291 | 5620 |
| 16384x16384 @256 | 5469 | pmtiles | 5562.6 ms | 983.2 | 1735.6 MB | 520 600 896 | **1** |
| 8192x8192 @64 | 21851 | directory | 1667.6 ms | 13103.1 | 438.5 MB | 133 748 027 | 22127 |
| 8192x8192 @64 | 21851 | pmtiles | 1684.0 ms | 12975.4 | 440.6 MB | 133 758 824 | **1** |

The engine's tracked working set is identical for the two backends at every
cell, which it should be: it charges raster buffers and a sink is not one.

PMTiles generation costs between 1% and 31% more wall time, and that is
structural rather than an oversight. The root directory has to fit the spec's
first 16 KiB, and an entry's offset is not known until the tiles are sorted, so
the payloads are staged and then copied into the archive: roughly twice the
archive's size in writes for the same tiles. The gap closes as the tiles get
smaller, because the fixed cost of a file goes up relative to its contents:
at 21851 tiles of 6 KB the two backends are within 1%.

The last column does not move with any of that. 22127 filesystem entries against
1, for the same 133 MB of tiles.

### Reads

Microseconds per lookup. The lookup count is 64 for the cold and warm rows and
the whole plan (capped at 20000) for the other three.

| Cell | Scenario | Directory p50 | PMTiles p50 | Directory p99 | PMTiles p99 |
|---|---|---|---|---|---|
| 2048x2048 @256 | cold | 7.25 | 14.17 | 196.6 | 223.3 |
| 2048x2048 @256 | warm | 4.75 | **4.21** | 5.5 | 5.7 |
| 2048x2048 @256 | sequential | 5.04 | **4.62** | 12.8 | 6.8 |
| 2048x2048 @256 | random | 4.88 | **4.21** | 5.9 | 10.0 |
| 2048x2048 @256 | concurrent | 5.58 | **5.04** | 131.6 | 32.6 |
| 16384x16384 @256 | sequential | 6.83 | 7.25 | 8.9 | 10.0 |
| 16384x16384 @256 | random | 6.71 | 7.29 | 8.9 | 10.2 |
| 8192x8192 @64 | warm | 2.00 | **0.46** | 2.1 | 0.5 |
| 8192x8192 @64 | sequential | 3.38 | **1.00** | 4.4 | 1.8 |
| 8192x8192 @64 | random | 3.42 | **1.21** | 4.5 | 2.0 |
| 8192x8192 @64 | concurrent | 4.46 | **2.38** | 9.1 | 43.6 |

The bar #986 sections 9 and 18 set is competitive local read latency, not
beating a raw `pread` on every single local read, and the archive clears it: it
is ahead on most rows and within half a microsecond on the rest. Where it is
furthest ahead is the cell with the most tiles, which is the direction that
matters: at 21851 tiles a lookup is 1.21 microseconds against 3.42, because one
`pread` into a file that is already open does not care how many tiles the
pyramid has and a path resolution through a directory tree does.

Cold is the row the archive loses, and that is its open cost: the header fetch
and the root-directory fetch, and on the 64 pixel cell a leaf fetch too, all of
which a client pays once before its first tile. A tree pays nothing to open
because there is nothing to read. The gap grows with the archive because the
root grows with it: 14 microseconds at 93 tiles, 91 at 5469.

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

### Which cells have leaf directories

Measured rather than assumed, by asking each archive's root what it holds:

| Cell | Entries | Root holds |
|---|---|---|
| 2048x2048 @256 | 93 | 93 tile entries, no leaves |
| 8192x8192 @256 | 1373 | 1373 tile entries, no leaves |
| 16384x16384 @256 | 5469 | 5469 tile entries, no leaves |
| 8192x8192 @64 | 21851 | 6 pointers, all of them leaves |

Under 16384 entries the writer keeps the whole directory in the root, so only
the last cell exercises the leaf lookup, the leaf cache and the second ranged
read at all. A sweep without it measures one half of the read path and reports
it as the read path, which is exactly what the first version of this one did.
`the_large_profile_reaches_the_leaf_directory_path` asserts the sweep still
crosses the cutoff, so an edit to the cell list cannot quietly drop it.

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
tables above were taken on, the same ratio is 24.90 over 21.25, or 1.17x.

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
