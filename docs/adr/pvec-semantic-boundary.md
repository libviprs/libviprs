# ADR: Normalized2DGeometry is a thin PVec adapter, and what may not be built here

Status: Frozen by N1.8 (#1047), except NG-12, which stays proposed (see Open items).
Decides N1.2 (#1041), measured against real entities by N1.3 (#1042), part of the epic #1039.
Evidence: [docs/spikes/dwg-mapping.md](../spikes/dwg-mapping.md). The full clause text, which I
did not copy, is [pvec-normalized-geometry.md](pvec-normalized-geometry.md), with its executable
checks in `tests/adr_normalized_geometry.rs`.

## Context

The risk the panel named (finding 1) is a second curve library growing in this crate one helper at a
time. N1.2 wrote the specification that forbids it and N1.3 tried it against 30 DWG fixtures.
Every numeric question is P1.2's and libviprs defines no epsilon of its own, so this ADR has no
threshold in it.

## Decision

- `Normalized2DGeometry` is a thin adapter over PVec's public semantic concepts and is not a second
  geometry library (NG-01). It holds values PVec's constructors accepted, hands them on and records
  what was done. It does not compute with them.
- No provider type, and no decoder IR type, appears in a field or a public signature of it (NG-08).
- Coordinates are source-space `f64`, unconverted (NG-04). `source_bounds` is PVec's conservative
  box, input for the spatial index only (NG-05).
- The prohibited constructs are NG-10's list, and a mapping that needs mathematics PVec lacks is
  `RequiresPVecCapability`, filed against `libviprs/pvec`, never worked around (NG-11).
- A curve is never tessellated here on the way into PVec (NG-V1).

## Numbers

N1.3 ran 30 fixtures (native x64 and arm64) and graded them against PVec at `d399d76`:

| grade | count |
|---|---|
| exact | 16 |
| equivalent | 8 |
| `RequiresPVecCapability` | 5 |
| refused | 1 |
| failed | 0 |

The five gaps are filed: bulge to arc (libviprs/pvec#91), ellipse from a major-axis vector and a
ratio (#92), periodic B-spline (#93, since closed by #98 after the pin), full-period trim canonical
form (#94). Of the NG-11 candidates, only the bulge and the ellipse constructor were real gaps. The
line segment, the arc bounds and the angle convention were not. N1.3 also settled the two open
items of N1.2: a closed polyline arrives as a flag with its first vertex copied, and PVec's bounds
exist for every kind and are conservative for splines.

## Consequences

- N3.1 implements the record of NG-02 and nothing larger. A reviewer rejects an added operation by
  pointing at NG-01 and the NG-10 row it matches.
- Day-one conversions will report `RequiresPVecCapability` for bulges and ellipses until PVec lands
  the constructors. That is a counted outcome, not a failure.

## Open items

- NG-12 (the element class and region side tables) is not frozen. It reserves a place against the
  issue text of N4.9 and N6.9, and the stream it feeds (libviprs/pvec#84, P3.9) is open on the PVec
  side. It needs a revision when that stream's version rules exist.
- The ADR pins PVec at `fdf87de`. N1.3 measured `d399d76` and N1.5 measured `1f7561e`, so the
  citations in NG-03 and NG-09 have to be re-checked in the change that moves the pin. I have not
  re-measured any mapping row against the later revisions.
- The mapping rests on `READER_MAPPING_ONLY` evidence: acadrust-written fixtures cross-read by
  ACadSharp, no vendor writer and no STEP oracle. A shared misreading of the DWG specification would
  not show. Not measured: 3D POLYLINE, polyface, a non-+z extrusion normal, a non-zero block base
  point, a SPLINE from fit points.
- Where derived entities take their `EntityKey` is N3.7's, and the decoder cannot yet deliver the
  identity that needs (#1239).
