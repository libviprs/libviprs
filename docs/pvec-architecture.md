# How a drawing becomes a PVec

This is the shape of the integration. Each stage has a page of its own, listed in the last section, and the decisions
behind it are the ADRs in `docs/adr/`. Every tolerance anywhere in it is `pvec_numeric::Policy::V1`
([frozen in the integration contract](pvec-integration-contract.md)); libviprs defines none.

```text
a DWG, or a vector PDF page
  -> provider                      the only seam a reader sits behind (DwgProvider)
  -> view profile                  which leaves are the visible drawing, and which were excluded and why
  -> block flattening              one composed map per leaf, bounded by the source limits
  -> extraction and promotion      one PVec curve per supported leaf, a class and a reason for the rest
  -> ordering                      dense file-local ids, deterministic
  -> the writer funnel             the artifact, written by PVec's own writer
  -> the conversion report         one row per entity met, checked against the count met
  -> atomic publish                artifact, report and georeference sidecar, all or none
```

## What libviprs is, and is not

libviprs is the policy layer between a source and PVec. It decides what is visible, what a transform does to an entity,
what the source's limits are and what to say about everything it did not carry. It does not compute geometry: the curves,
the bounds, the promotion rules and the container are PVec's, and `Normalized2DGeometry` is a thin adapter over PVec's
public types (`docs/adr/pvec-semantic-boundary.md`). A mapping that needs mathematics PVec lacks is reported as
`RequiresPVecCapability` and filed against PVec, never worked around here.

## Decisions that shape everything else

- **Source space, always.** PVec coordinates are the drawing's own units, never a projection. Placement on a map is a
  sidecar (`docs/pvec-georeference.md`).
- **Nothing is tessellated and nothing is repaired.** An entity PVec cannot carry is counted with a reason code and left
  out. A display-only result is never called exact.
- **Every entity is counted once.** The report's rows equal the entities met, and the class counts add up to the same
  number or the conversion fails.
- **One writer funnel.** A PVec writer call anywhere else under `src/` turns `tests/pvec_api_surface.rs` red.
- **Deterministic.** Two conversions of one drawing publish the same bytes, at any queue capacity.
- **Open source only.** The default provider is acadrust, `acadsharp-rs` is opt-in, and there is no LibreDWG or other
  proprietary dependency of any kind (`docs/pvec-providers.md`).

## What is not built

The whole-conversion spool (the writer's spill is built, #1314 and #1417, but the classifier's and the report's own state is not), the conversion's call to the component and provenance streams (#1093) and to the derived room regions
are not built (the functions exist, `pvec::components` and `pvec::regions`). Peak memory of the in-memory path is therefore linear in the entity count: about 2.8 KB per entity measured, which misses the
1.5 GiB ceiling at a million entities (`docs/pvec-benchmarks.md`). Every page below says where this touches it.

## The pages

[Provider API](pvec-provider-api.md), [ISO mapping](pvec-iso-mapping.md), [view semantics](pvec-view-semantics.md),
[transform matrix](pvec-transform-matrix.md), [DWG support](pvec-dwg-support.md), [PDF loss model](pvec-pdf-loss.md),
[components](pvec-components.md), [georeference handoff](pvec-georeference.md), [conversion API](pvec-conversion-api.md),
[limits](pvec-limits.md), [provider replacement](pvec-provider-replacement.md), [benchmarks](pvec-benchmarks.md),
[version compatibility](pvec-version-compatibility.md), [the frozen contract](pvec-integration-contract.md).
