# ADR: Provenance is optional component data outside the geometry hot path

Status: Frozen by N1.8 (#1047). Decides the provenance clause (NG-06) of N1.2 (#1041), part of the
integration epic #1039. The payload shape is N3.2's (#1096).
Evidence: [docs/spikes/view-context.md](../spikes/view-context.md) (what the decoder boundary
carries today) and [docs/spikes/dwg-mapping.md](../spikes/dwg-mapping.md). Full clause text in
[pvec-normalized-geometry.md](pvec-normalized-geometry.md).

## Context

Provenance (source handle, layer, block lineage, provider) is useful for reports and for explaining
a refusal, and expensive if it rides inside every geometry record. The numeric policy is P1.2's and
libviprs defines no epsilon of its own, so nothing here is a threshold.

## Decision

Provenance is optional component data, reachable from a record through an opaque handle and held
in its own store, never inside the mathematical entity (NG-06):

- A record with provenance absent is complete. Every consumer in N6 and N7 works on `None`.
- No equality, ordering or hashing of a record reads the handle.
- Dropping the store changes no geometry byte, and an output stream with the provenance component
  skipped is valid.
- Derived entities (gap-closing segments, flattened inserts) say they are derived in their
  provenance component, never as a geometry field.

## Numbers

What the decoder boundary carries today, measured in N1.4 on native x86_64: a geometry record has an
item handle and a flattened-from-insert flag, and nothing else. No layer name, no block name, no
insert chain. A leaf's handle repeats once per instance: `4C` twice in `vc_nested` with different
coordinates, `52` three times in `vc_layers`. The handle is therefore not an instance identity, so
the lineage the provenance store holds has to come from the flattener and cannot be recovered
from the handle. The gap is #1239.

## Consequences

- N3.2 defines the payload and the store, N3.7 the entity key that links to it.
- Keeping provenance out of the record means the hot path (ingest, transform, index) never touches
  it, and a conversion can run with provenance disabled.

## Open items

- The "outside the hot path" claim is structural. No benchmark measured the cost of carrying
  provenance inline against a side store, so the size of the saving is unmeasured. N8.3 and N8.4 are
  where that measurement belongs, and the decision does not rest on it.
- The payload cannot be filled from the decoder boundary as it is (#1239), so lineage and layer
  provenance wait on a wire change, a side table or the acadrust provider.
- Instance identity across a MINSERT or a repeated block is not yet defined (N3.7).
