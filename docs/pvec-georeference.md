# Georeference handoff

PVec coordinates are always source space. A conversion never writes a transform, a CRS string or a projected coordinate into the artifact.
What an application needs to place the drawing on a map goes in one small sidecar beside it, `<dest>.georef.json`, and the conversion report
names it. The decision is in the module docs of `src/pvec_convert/georef.rs`.

## The sidecar

Versioned like the other manifests: `schema_version` is the tag, and a file with another version is refused (`UnknownVersion`), never
parsed leniently.

| field | meaning |
|---|---|
| `bounds` | the exact drawing box in source units, the one the artifact's manifest declares; absent for a drawing with no geometry |
| `unit.source` | the unit the drawing's header declares, `Unknown` when it declares none, never defaulted |
| `unit.insunits` | the header's `$INSUNITS` code, when there is one |
| `unit.physical` | whether the source unit is a real length unit |
| `unit.evidence` | what the source unit was read from, in words |
| `unit.applied` | the unit the artifact declares; differs from `source` only when the caller overrode it |
| `calibration_id` | an opaque identifier of a calibration record the application holds |
| `control_points` | source-space points, drawing coordinates only, with no world coordinate attached |

## Re-calibrating

`GeoreferenceHandoff::recalibrate` rewrites the sidecar and touches nothing else, so the artifact's bytes and checksum do not move.

## Units

A unit PVec cannot declare is refused by the writer, not rescaled (rescaling would change every coordinate). A drawing with no unit needs
one from the caller (`PvecConversion::unit`). See `docs/pvec-units.md`.
