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
- The spline extractor refuses over PVec's own counts (`MAX_DEGREE`, `MAX_CONTROL_POINTS`, `MAX_KNOTS`).
- A recursion bomb is refused by the depth limit under both the fuzz target and a direct test.

## How they are tested

`tests/pvec_1101_source_limits.rs` for each limit, `tests/pvec_1429_bench_corpus.rs` for the `malformed` benchmark cell (one input per
limit, each refused by the limit it names), `tests/pvec_1439_view_limits.rs` for the view walk, and the fuzz targets `fuzz_pvec_provider`,
`fuzz_pvec_spline` and `fuzz_pvec_clip` under tight limits. The PDF vector path has its own `PdfVectorLimits` (`docs/pvec-pdf-loss.md`).

## What is not bounded yet

Memory, in the large. The limits bound what a single drawing may ask for, not what a conversion holds: peak memory is linear in the
entities until the spool exists (`docs/pvec-benchmarks.md`).
