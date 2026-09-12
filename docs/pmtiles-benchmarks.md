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

Reads report five scenarios. `read_cold` is one lookup on a reader that has
never been used, `read_warm` is the same lookup again on the same reader,
`read_sequential` walks the plan in order, `read_random` walks a deterministic
shuffle, and `read_concurrent` runs the same shuffle across every available
core.

**Cold means a cold reader, not a cold page cache.** Dropping the OS page cache
needs root on Linux and has no portable equivalent, so the harness does not
claim to have done it. The gap between the cold and warm rows is the
in-process caches alone (the PMTiles leaf cache, the directory reader's lack of
one), which is the part libviprs controls and the part an optimisation would
move.

## Running it

The cheap profile is the default and takes seconds:

```sh
cargo test --release --test pmtiles_benchmarks -- --ignored --nocapture
```

The large profile walks three canvases up to 16384x16384 and takes minutes. It
is opt-in on purpose, since a benchmark nobody runs because it is too expensive
is a benchmark nobody runs:

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
