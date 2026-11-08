# Georeference handoff

PVec coordinates are always source space, and no transform is ever applied to them. Since #1552 a drawing's calibration (a scale, a
georeference or both, `pvec_convert::Calibration`) lives inside the artifact, in PVec's calibration block (capability 22, block kind 11,
pvec ADR 0019). A DWG gets one only when the caller passes it to `PvecConversion::calibration`; a PDF gets what its page says
(`docs/pvec-conversion-report.md`, `pdf.calibration`). `pvec_convert::recalibrate` adds, replaces or removes it later.

The `<dest>.georef.json` sidecar below is still written, with the calibration's `external_id` as its `calibration_id`, until it is
retired (the last slice of #1543).

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

`pvec_convert::recalibrate(path, Some(&calibration))` replaces the calibration block through PVec's rewrite, and `None` removes it. Every
other block keeps its bytes and its checksum; the file's SHA-256 is recomputed, and the new file is renamed over the old one, so a reader
sees one or the other. Putting back the calibration a conversion wrote gives back the bytes it wrote. `GeoreferenceHandoff::recalibrate`
still rewrites the sidecar alone.

## Units

A unit PVec cannot declare is refused by the writer, not rescaled (rescaling would change every coordinate). A drawing with no unit needs
one from the caller (`PvecConversion::unit`). See `docs/pvec-units.md`.
