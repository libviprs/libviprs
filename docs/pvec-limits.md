# Resource and security limits

A drawing is hostile input. The limits below bound a conversion the way the crate already bounds its other readers. They are
`SourceLimits` (`pvec::limits`), carried by the view profile, and charged by the view walk and by the block flattener. Every breach
names the limit and the observed value; nothing panics and nothing truncates.

The defaults are backstops against crafted input, not tight bounds: a real drawing, including a large site plan, stays far below
all of them.

<!-- limits:begin -->
| limit | default |
|---|---|
| `max_block_depth` | 64 |
| `max_entities` | 20000000 |
| `max_polyline_vertices` | 1048576 |
| `max_spline_control_points` | 1048576 |
| `max_spline_knots` | 1048640 |
| `max_insert_grid` | 1000000 |
| `max_expanded_instances` | 10000000 |
| `max_allocation_bytes` | 2147483648 |
| `max_name_bytes` | 1020 |
<!-- limits:end -->

Raise one with `SourceLimits` when a legitimate input trips it, or use `SourceLimits::raised()` for a trusted source.

## What enforces them

- `ViewProfile::apply` charges every entity it visits and the expanded instance total, so a block that fans out is refused before it
  is expanded (#1439).
- `BlockResolver::flatten` enforces depth, cycles, the grid, the expanded total and the allocation estimate, and a per-block entity ceiling.
- A sheet under the print profile checks `max_entities` against what it will classify: its own items, plus the kept model leaves
  times the viewports drawing them, plus the model's excluded and skipped rows once. The sheet and model space are each walked under
  their own guard, so the product is checked before any of it is listed, and a breach is the flattener's `limit` failure (#1580).
- The spline extractor refuses over PVec's own counts (`MAX_DEGREE`, `MAX_CONTROL_POINTS`, `MAX_KNOTS`).
- A recursion bomb is refused by the depth limit under both the fuzz target and a direct test.

## How they are tested

`tests/pvec_1101_source_limits.rs` for each limit, `tests/pvec_1429_bench_corpus.rs` for the `malformed` benchmark cell (one input per
limit, each refused by the limit it names), `tests/pvec_1439_view_limits.rs` for the view walk, and the fuzz targets `fuzz_pvec_provider`,
`fuzz_pvec_spline` and `fuzz_pvec_clip` under tight limits. The PDF vector path has its own `PdfVectorLimits` (`docs/pvec-pdf-loss.md`).

### Clean fuzz time (#1070)

The acceptance bar is 60 seconds per target. On top of that I ran each of the four conversion targets for 20 minutes, on mercury (aarch64, native), in a `rust:1.99.1-bookworm` container with nightly `cargo fuzz` (AddressSanitizer on, 3 GiB rss limit), starting from a copy of the committed seed corpus, at libviprs main 97df887c:

| target | runs in 1,201 s | result |
|---|---|---|
| `fuzz_pvec_provider` | 632,756 | no crash |
| `fuzz_pvec_spline` | 12,202,522 | no crash |
| `fuzz_pvec_clip` | 8,198,203 | no crash |
| `fuzz_pvec_spool` | 87,172 | no crash |

That is 80 minutes of clean fuzz time in total, 20 minutes per target, on arm64 only. I took no x86_64 fuzz run this time because the NAS was carrying the pre-commit hooks of other jobs for the whole window. It is minutes per target, not a soak run, and the out-of-scope line for multi-hour soaks stands.

## What is not bounded yet

Memory, in the large. The limits bound what a single drawing may ask for, not what a conversion holds: peak memory is linear in the
entities of the in-memory path. The writer's spill (`PvecConversion::spill`, #1314 and #1417) moves the writer's share out of memory, but the whole conversion was re-measured with it (`docs/pvec-benchmarks.md`, #1064) and the peak did not move: libviprs' own per-entity state is what sits in memory. #1481 cut the tracked peak at a million entities from 2,015 to 1,323 MiB and restated the criterion as a linear budget (`docs/pvec-benchmarks.md`, "After the stack").
