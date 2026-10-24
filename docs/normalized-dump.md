# Normalized dump (N3.8, issue #1103)

A text dump of `Normalized2DGeometry` records, for working out whether a bad conversion is a
normalization problem or a PVec problem. `libviprs::pvec::dump` (feature `pvec-normalized`)
writes it. There is no binary and no new dependency: call `NormalizedDump` from a test or from
debug code, or `dump_to_string` for a one-liner.

## This is not an interchange format

It is inspection output only. It is not an interchange format, it has no schema guarantees beyond
the version line, and no loader consumes it: nothing in libviprs, PVec or the tests reads it back,
and nothing should. If you find yourself wanting to parse it, you want a PVec artifact instead.
Reading it back would make it a second serialization of the geometry, which is the thing it is
deliberately not.

## Independent of PVec writing

The dump is produced at the normalization stage. `NormalizedDump` writes each entity into its
sink as soon as it is pushed, so when a conversion fails later, in PVec writing, the sink already
holds every entity normalized before the failure. A dump that was never finished is recognisable
by its missing `end` line.

## Format (version 1)

Line oriented, UTF-8, `\n` line endings, fixed field order, so two dumps of the same input diff
cleanly and are byte-identical.

```
# libviprs normalized dump v1
# inspection output, not an interchange format, nothing reads it back
entity <id> <type> <fidelity>
  geometry <type> <key>=<value> ...        (or: geometry none)
  bounds min=(<x>,<y>) max=(<x>,<y>)       (or: bounds none)
  provenance source=".." layout=".." handle=<hex>|- class=".." layer=".." path=<block@hex/...>|/ xf=[m00, m01, m10, m11, tx, ty] warnings=[..] [truncated]
end entities=<n>
```

- `<id>` is the file-local entity ID. `NormalizedDump::push_with_id` prints the `EntityId` that
  `order_entities` (N3.7) assigned, so a dump line and a conversion report line name the same
  entity. `push` numbers by position in push order, from 0, for callers that have no ordering
  step. Pick one per dump.
- `<type>` is one of `line`, `circle`, `ellipse`, `polyline`, `trimmed`, `composite`, `bezier`,
  `bspline`, `rational-bspline`, or `none` for a record that holds no geometry
  (`RequiresPVecCapability`, `Unsupported`, `IgnoredByPolicy`).
- `<fidelity>` is the N3.5 class tag. Every entity has one, and the `end` line's count is the
  number of entities written, which equals `FidelityReport::encountered()` when every
  encountered entity was pushed, no-geometry ones included.
- Parameters are the source-space values PVec holds, unconverted. Directions are PVec's
  normalised ones. A trimmed curve prints `start`, `span` and `reversed` (PVec's normalised trim)
  and nests its basis in parentheses.
- Every number is `f64`'s shortest round-trip text, so `parse::<f64>()` on a printed value gives
  back the identical bits.
- The `provenance` line appears only when the caller supplies a record (block path, layer, handle
  and the transform as extracted). With provenance off, that line is absent and every other line
  is unchanged.

## Feeds

N2.3 stage attribution, N6.7's conversion report and N7.5's structured diagnostics can point at a
dump line. None of them parse it.
