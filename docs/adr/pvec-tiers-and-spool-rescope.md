# ADR note: tiers are PVec's, and the writer's memory is linear (re-scope of N6.5 and N6.3)

Status: **Proposed, awaiting Rom.** Nothing below changes `#1064`'s criteria or drops a line of `#1314`
until he decides. Part 1 (tiers, decision D8) only narrows what `#1314` asked for to what PVec's
writer now does, and is implemented. Part 2 (memory, decision D1) is a proposal.

Refines `pvec-streaming-strategy.md` and the N6.3 and N6.5 text consolidated into #1314.

## Part 1: N6.5, progressive tiers (D8)

### What changed under the issue

N6.5 was written when libviprs would have to tell PVec which entities are cheap enough to draw
exactly and which need a preview, through a `TierAssignment` per entity in `src/pvec/tiers.rs`.
PVec's facade has since grown the indexed writer:

```text
PvecWriter::new_indexed(manifest, cell_capacity, tier0_max_error)
```

It lays the spatial directory (P5.1) and one Tier 0 block per leaf (P5.4b) itself. Which entity is
exact in Tier 0 and which gets a bounded-error preview is decided inside PVec's `Tier0Builder` from the
curve and the bound, and the preview mathematics and the error bound are PVec's. A `TierAssignment` in
libviprs would be a second copy of that decision.

### Decision

* No `src/pvec/tiers.rs`, no `TierAssignment`, no preview request type in libviprs.
* The funnel (`src/pvec/writer.rs`) passes PVec the two numbers `new_indexed` takes, as
  `LeafConfig { cell_capacity, tier0_max_error }`. Both come from the caller. libviprs defines no
  tolerance, so there is no numeric tolerance literal in the funnel (a test over its source holds that).
* The report (`#1093`, N6.7) records preview coverage as display fidelity, read back from PVec's
  Tier 0 leaves, and never gives a preview a geometric-exactness class.

### The N6.5 acceptance criteria, re-read

| criterion | how it holds now |
|---|---|
| one tier assignment per entity | PVec assigns, one Tier 0 entry per entity id; the test reads the leaves back and counts |
| canonical geometry for a deferred entity is still the analytic curve | `tests/pvec_1314_writer_funnel.rs` reads the indexed artifact back and compares the curve kinds |
| every preview resolves to one canonical entity | PVec's `P5.4` replacement identity; libviprs holds no preview |
| no tolerance literal in `tiers.rs` | there is no `tiers.rs`; the same check runs over `writer.rs` |
| the report labels preview coverage as display fidelity | `#1093`, from the Tier 0 leaves |

This does not contradict `#1092`'s intent (a complete drawing before every curve is refined), only
where the decision lives. Rom only needs to look at it if he wants libviprs to keep its own tiering
policy, which I would argue against.

## Part 2: N6.3 and the 1.5 GiB criterion (D1), decided

### The measurement

The facade's writer keeps every push until `finish` (the indexed one keeps an operation log to lay
the file out a second time) and `finish` returns the artifact as one `Vec<u8>`. A spool on
libviprs's side cannot bound that, because the entities have to reach the writer in memory anyway.

Measured on MARS (native x86_64, `rust:1.99.1`), two-point polylines, pvec at `b9f4416`:

Through the funnel (`examples/pvec_writer_memory.rs`, release build, MARS native x86_64,
`rust:1.99.1`, two-point polylines). "Added" is the peak resident set during `write_geometry` minus
the resident set before it, so it excludes the caller's own geometry (about 440 bytes an entity):

| entities | layout | added peak | per entity | artifact |
|---|---|---|---|---|
| 1,000,000 | plain | 149.6 MB | 157 B | 2.86 MB |
| 1,000,000 | indexed, Tier 0 | 391.7 MB | 411 B | 4.10 MB |
| 4,000,000 | plain | 596.4 MB | 156 B | 11.4 MB |
| 4,000,000 | indexed, Tier 0 | 1,521 MB | 399 B | 17.4 MB |

Linear in the entity count, and the indexed writer costs about 2.6 times the plain one. Earlier,
against the bare facade (pvec at the same revision), the audit measured 312 MB peak at 1M and
1,239 MB at 4M.

### Decision

Rom chose to stream in pvec first (libviprs/pvec#235: `finish_to(impl Write)` with an external-sort
layout, byte-identical to `finish()`). So N6.3 and the 1.5 GiB criterion of `#1064` stay as written,
and the bound comes from PVec, not from a libviprs spool.

* No libviprs spool code is written now. N6.3 closes only after pvec#235 merges, the pin moves, and
  the numbers from `examples/pvec_writer_memory.rs` meet the criterion.
* The funnel today returns `Vec<u8>`. When #235 lands it gains a sink variant that calls
  `finish_to`, and `publish` takes the stream instead of a slice, writing straight into the
  `.pvec.partial` file. The cancel token, the stages and the cleanup do not change.
* Until then `#1064` N8.4 and N8.6 stay blocked, and everything else in `#1314` carries on.
