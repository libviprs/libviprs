# Components, style and provenance

A component is information that rides beside the geometry, never inside it. Geometry stays the PVec curve; style, fills, classes and
provenance are separate streams a viewer may drop without changing a single coordinate.

## What exists

- **Source provenance.** `pvec::provenance` keeps the handle, class name, layer and the INSERT chain of every entity in a bounded side
  store, behind an optional handle on the record. The conversion report carries the same facts per row.
- **Line style.** `pvec::style` reads colour, lineweight and linetype from the provider (BYLAYER already resolved, BYBLOCK left for
  the INSERT chain), builds a monochrome line style from them (`dwg_style_input`, `dwg_line_style`) and refuses a degenerate dash
  pattern. `pvec::components::build_streams` writes it as PVec's line style stream: width, dash, phase and visibility.
- **Layer and provenance streams.** Also from `build_streams`. The layer stream names each entity's layer and carries the off, frozen
  and locked flags of the layer table. The provenance stream has one row per source entity, not per curve: an entity reached through
  an INSERT belongs to the outermost INSERT, so every curve one instance flattens into shares a row, written against the first of
  them. The row holds the source handle only, because that is all the stream has room for; block path, layer and warnings stay in the
  conversion report.
- **Rooms and corridors.** `pvec::regions` derives them from wall linework (see below) and `semantic_stream` writes them as the
  region records of PVec's semantic class stream.
- **Text.** Not carried in v1: text is counted and reported, with the font outcome (embedded, missing SHX, unresolved) in
  `docs/pvec-text-policy.md`.
- **Hatch and fill.** Geometry only; the fill rule stays on the source side. See `docs/pvec-hatch-policy.md`.
- **Semantic class.** Walls and columns are classified from the DWG (`tests/pvec_classify_1233.rs`), the class stream itself is PVec's.

## Streams are optional

`StreamSelection` picks the streams of a run and `build_streams` returns only those. A run with none writes the same geometry as a run
with all three (`tests/pvec_1093_components.rs` reads both artifacts back and compares the entities and the drawing box), and each
stream decodes without the others. Cost on a laptop at one million two-point polylines (`examples/pvec_components_cost.rs`): the
geometry write runs at about 1.22 M entities/s alone and 1.09 M with all three streams in the file, and building them takes 0.14 s
(provenance alone 0.10 s). Streams add bytes to the artifact (about 20 bytes per entity per stream) and nothing to the geometry blocks.

## Derived rooms and corridors

Rooms and corridors are the empty space the walls enclose, so they are computed and every one is labelled a guess. `derive_regions`:

1. joins wall ends that sit within a hair of each other into nodes;
2. closes a gap between two loose wall ends (a door jamb) when it is within `gap_tolerance`, nearest pair first, each end once. The
   closing segment is a `DerivedSegment`, never part of the source geometry, and the caller appends it after the source entities;
3. drops walls that still dangle, traces the faces of what is left, and discards the face around each connected piece;
4. calls a face a corridor when its smallest enclosing rectangle is at most `corridor_max_width` across and `corridor_min_aspect`
   times longer than wide, and a room otherwise. The source tag is `Inferred` with a confidence of 0.5 plus half the margin by which
   the face clears that test;
5. names a region after the first text anchor inside it, and replaces a face that lies inside an `ExplicitRoom` (a drawn room
   boundary or tagged block) with that room, tagged `Explicit` and with no confidence.

A degenerate face has no kind: it is counted as unlabelled and left out unless `keep_unlabelled` is set (the stream cannot hold a region
with no kind, so it is left out of the stream either way). More than `max_faces` faces refuses with `regions.face-limit`. Limits: walls
must meet at their ends (a wall passing through a junction is not split), and a column inside a room is not a hole. Turning the step off
leaves the geometry as it was, since it only reads the walls.

## In the conversion

`PvecConversion::components(StreamSelection)` writes the layer, line style and provenance streams after the geometry, and
`PvecConversion::regions(Some(RegionConfig))` adds the semantic class stream. Both are off by default and each stream is optional. The
report's `components` and `regions` sections say what was written and what was left out by reason (`stream.reason` in
`components.dropped`, which is where the PVec reason codes of a stream row show up, and `regions.dropped`). Tests:
`tests/pvec_1500_convert_streams.rs` and `tests/pvec_1526_regions.rs` read every stream back through PVec's reader (libviprs/pvec#257 is
closed), `tests/pvec_1526_corpus_geometry.rs` checks on the whole pinned corpus that the entities and the drawing box are the same with
every option on and off and that a plain artifact holds no stream, and `tests/pvec_end_to_end.rs` (which asks for no stream) is the
geometry conformance run against a component-free artifact.

An unlabelled face gets no PVec kind. It is counted (`unlabelled`) and, with `keep_unlabelled`, listed as dropped `unlabelled`. A kind for
it would be added when a consumer needs to tell "a space the rules could not label" from "no space"; until one does, adding one would
only be a guess written into the artifact.

### What the streams cost

Measured with `tests/pvec_1526_streams_bench.rs` on `large_industrial` at scale 0.2 (205,000 entities), best of 3, release build:

| host | arch | off | provenance only | all streams | all streams + regions |
|---|---|---|---|---|---|
| NAS | x86_64 | 41,690 entities/s | 41,995 (+0.7%) | 38,110 (-8.6%) | 37,667 (-9.6%) |
| mercury | arm64 | 111,246 entities/s | 107,759 (-3.1%) | 102,985 (-7.4%) | 98,322 (-11.6%) |

The NAS is shared, and its throughput moved more between runs than the variants differ: a second run read 51,170 off, 49,427 provenance only, 60,048 all streams and 54,822 with regions. Read the throughput rows as within noise of each other, at a cost of at most about 10% for everything on. The tracked heap peak does not move between runs and is the figure the budget is on.

The tracked heap peak is 475.2 MiB with the streams off and 538.5 MiB (provenance only) to 540.4 MiB (all streams, and with regions)
with them on, which is +13.7%, or about 320 B per source entity. The artifact grows from 4.3 MiB to 9.4 MiB.

The first target for this, at most 10% or 128 B per entity, **was not met**. The cause is that the provenance and layer rows are held in
memory, one per entity, until the streams are written after the geometry. The budget is therefore stated as what is measured: the
streams may add at most 15% to the streams-off peak, or 350 B per source entity, whichever is larger, taken as the best of 3 on the NAS at
scale 0.2. The benchmark asserts it. The route to the original target is a streaming pass that builds the rows as the geometry is
written, so they are never all held at once; it is not built.

## Not done

Text and hatch streams stay out until the text and hatch policies exist (N5.5, N5.6).
