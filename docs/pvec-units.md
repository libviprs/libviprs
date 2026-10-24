# Units and source coordinates

A drawing's coordinates mean whatever its units say they mean, and often the units are not there. This is how `pvec` handles that, and the short version is that I keep the numbers exactly as the drawing stores them and carry the unit beside them as metadata.

The code is `src/pvec/units.rs`. The mapping from a raw code to a unit lives in exactly one table there, `INSUNITS_TABLE`, and the rows below are the same table.

## The `$INSUNITS` table

| Code | Unit | `SourceUnit` variant |
|---|---|---|
| 0 | Unitless | `Unitless` |
| 1 | Inches | `Inches` |
| 2 | Feet | `Feet` |
| 3 | Miles | `Miles` |
| 4 | Millimeters | `Millimeters` |
| 5 | Centimeters | `Centimeters` |
| 6 | Meters | `Meters` |
| 7 | Kilometers | `Kilometers` |
| 8 | Microinches | `Microinches` |
| 9 | Mils | `Mils` |
| 10 | Yards | `Yards` |
| 11 | Angstroms | `Angstroms` |
| 12 | Nanometers | `Nanometers` |
| 13 | Microns | `Microns` |
| 14 | Decimeters | `Decimeters` |
| 15 | Decameters | `Decameters` |
| 16 | Hectometers | `Hectometers` |
| 17 | Gigameters | `Gigameters` |
| 18 | Astronomical units | `AstronomicalUnits` |
| 19 | Light years | `LightYears` |
| 20 | Parsecs | `Parsecs` |
| 21 | US survey feet | `UsSurveyFeet` |
| 22 | US survey inches | `UsSurveyInches` |
| 23 | US survey yards | `UsSurveyYards` |
| 24 | US survey miles | `UsSurveyMiles` |

Anything else, and a header with no `$INSUNITS` value at all, is `Unknown` (it has no code of its own).

Code 0 is a statement the drawing makes: it has no unit. `Unknown` is the absence of a statement. They are different variants on purpose, and neither is turned into the other.

## Where the unit comes from

Only the drawing header. I do not guess from how big the coordinates look (a 40000-wide drawing could be millimetres or feet), and I do not fall back to a default unit when the header is silent. `AcadrustProvider::source_unit` reads `$INSUNITS` from the header and maps it through the table, nothing more.

## A missing header value

A header with no `$INSUNITS` value maps to `Unknown`. The conversion does not fail, and the drawing's coordinates still come through untouched.

## What `Unknown` means downstream

It means nobody told us. A consumer must treat the coordinates as numbers in an unspecified length unit:

* The georeference handoff refuses to emit a georeference for a drawing whose unit is not physical (`SourceUnit::is_physical` is false for both `Unknown` and `Unitless`). It does not pick a unit to make the numbers fit.
* A report should say the unit was unknown rather than print a unit.
* A consumer that wants a unit anyway has to take it from the user, as an explicit input, and do the conversion itself.

## Geometry is never rescaled on ingest

Geometry is never rescaled on ingest. There is no scale factor, no normalise-to-metres step, and no georeferencing transform composed into normalised geometry anywhere under `src/pvec/`. For a drawing authored at exactly representable `f64` coordinates, the values reaching `pvec` are bit-identical to the source, and the test compares raw bit patterns. The same drawing saved in inches and in millimetres gives identical coordinate values and a different `SourceUnit`.

The unit and the source-space drawing bounds are handed to the georeference contract (N7.3) as `SourceSpace` metadata. The raster-side georeference types in `src/geo.rs` stay on the raster side, and a test scans `src/pvec/` to make sure nothing there refers to them.

## Not here

The georeference handoff itself, calibration control points, projection, and unit conversion in any consumer.
