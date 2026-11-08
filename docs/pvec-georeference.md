# Placing a drawing: the calibration block

PVec coordinates are always source space, and no transform is ever applied to them. What places a drawing in the world is
its calibration, `pvec_convert::Calibration`, which lives inside the artifact in PVec's calibration block (capability 22,
block kind 11, pvec ADR 0019). It holds a scale, a georeference or both, and the application's own id for it:

| record | what it says |
|---|---|
| scale (`ScaleClaim`) | what one source unit stands for in the world, in metres, and where that came from: a unit the manifest cannot name (a PDF's point), a plan scale stated on the sheet, two points and a measured distance, the georeference, or typed in by hand |
| georeference (`Georeference`) | an affine from source space into a CRS (`[a b c d tx ty]`), the CRS's kind, its authority code or its definition (WKT1, WKT2 or PROJJSON, never parsed here), how the affine was arrived at, the control points behind it and the fit's largest residual |
| `external_id` | the application's id for the calibration |

The manifest's unit says what one coordinate unit is; the scale says what it stands for. They differ on any drawn-to-scale
sheet, and neither overwrites the other. `Calibration::new` refuses what the block would refuse, and a projective
transform is refused with `CalibrationError::Projective`, since the block holds an affine only.

## Where a calibration comes from

- **A DWG** gets one only when the caller passes it to `PvecConversion::calibration`. Its `$INSUNITS` is already the
  manifest's unit (`docs/pvec-units.md`).
- **A PDF** always gets a scale: 0.0254 / 72 m per point (times `/UserUnit`), or the plan scale a `/VP` viewport's `/RL`
  measure states. A GeoPDF viewport (`/Measure /Subtype /GEO`) adds a georeference fitted to its control points, and
  without one an `/LGIDict` in UTM on WGS 84 adds a declared one. The report's `pdf.calibration` says what was used and
  why anything was not (`docs/pvec-conversion-report.md`).

## Re-calibrating

`pvec_convert::recalibrate(path, Some(&calibration))` adds or replaces the block through PVec's rewrite, and `None` removes
it. Every other block keeps its bytes and its checksum; the file's SHA-256 is recomputed, and the new file is renamed over
the old one, so a reader sees one or the other. Putting back the calibration a conversion wrote gives back the bytes it
wrote.

## The retired sidecar

Until #1554 a conversion wrote `<dest>.georef.json` beside the artifact: the drawing box, the unit, an opaque
`calibration_id` and source-space control points with no world side. No conversion writes it now. For one minor release
an old one can still be brought in:

```rust
let done = import_sidecar(&artifact, &georef_path(&artifact), |id| my_store.calibration(id))?;
```

The resolver turns the sidecar's `calibration_id` into a calibration, which is written into the artifact with that id as
its `external_id`. A sidecar with no id gives `SidecarImport::NoCalibrationId`, an id nothing resolves gives
`SidecarImport::Unresolved`, and in both cases nothing is written. `GeoreferenceHandoff` (the reader) is deprecated, and
goes with `import_sidecar` in the next minor release. `PvecConversion::control_points`, whose points only ever reached
the sidecar, is already gone (#1556): control points go in a `Georeference` passed to `calibration`.
