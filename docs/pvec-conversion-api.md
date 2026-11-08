# The conversion API

`libviprs::pvec_convert::PvecConversion` (feature `pvec-convert`, which joins `pvec-write` and `dwg-acadrust`) is the one surface an
application calls. It is built the way `EngineBuilder` is.

```rust
let out = PvecConversion::new(CadSource::Path(&dwg), &dest)
    .unit(SourceUnit::Meters)          // when the drawing declares none
    .profile(ConversionProfile::Default)
    .convert()?;
assert_eq!(out.report.totals.encountered, out.fidelity.encountered());
```

## Builder

`view_profile`, `limits`, `unit`, `leaves` (spatial directory and Tier 0), `cancel`, `observer`, `queue_capacity`, `calibration` (a `Calibration`, written into the artifact's calibration block),
`profile`, `allowed_capabilities`, `capability_policy`. `probe` returns the per-class counts without writing, `convert`
writes. `convert_with::<P>` and `probe_with::<P>` take any `DwgProvider`; the trait carries no unit, so `unit` is required there.

## Profiles and the capability gate

The caller says before the run which PVec capabilities it may use. An entity needing one outside the set is refused, never reshaped.

| profile | capabilities | entities | policy |
|---|---|---|---|
| `Strict` | those of the structurally exact rows | `SupportedExact` only | `RejectInput` |
| `Default` | every row of the mapping table | exact and equivalent | `RejectEntity` |
| `Permissive` | every row of the mapping table | exact and equivalent | `Downgrade` |

`RejectInput` aborts with `CapabilityNotAllowed` naming the capability and the entity. `RejectEntity` leaves the entity out and counts it as
`RequiresPVecCapability`, reason `capability-not-allowed`. `Downgrade` has no weaker representation to fall back to in v1 (none is not a
tessellation), so today it records `downgrade-unavailable` and omits the entity.

## Output and errors

`PvecConversionOutput` has the published artifact path, the report, the independent per-class counts, the entities written and the peak queue
depth. `PvecConvertError` keeps every lower error as `source()`: open, view, flatten, unit required, write, tally, report, publish, georeference
(a deprecated control point that is not finite), capability not allowed, cancelled. After any error the destination and its report are as they
were.

## Progress, cancellation, backpressure

The classifier and its consumer meet in a bounded queue (default 64, `queue_capacity`), the observer hears the stages as `EngineEvent`s, and the
cancel token is polled once per batch of 256 entities and during the write. A cancel that fires during a spool flush stops the write there (#1463).

## What it writes

The artifact (with its calibration block when there is a calibration, `docs/pvec-georeference.md`) and `<dest>.report.json`
(`docs/pvec-conversion-report.md`), published together or not at all. No georeference sidecar is written since #1554. Diagnostics have stable codes (`docs/pvec-diagnostics.md`).

## Every sheet of a drawing

`PvecLayoutSet` (#1564) converts each paper layout to its own artifact, one call for the drawing. It opens the drawing once, walks the paper
layouts in tab order and runs the conversion above under `ViewProfile::print_profile(layout)` for each.

```rust
let set = PvecLayoutSet::new(CadSource::Path(&dwg), &out_dir)
    .layouts(["A-101", "A-102"])           // optional; an unknown name fails the whole set
    .sheets(|c| c.components(StreamSelection::ALL))
    .convert()?;
for s in set.failed() { eprintln!("{}: {:?}", s.layout, s.outcome); }
```

Each sheet is `<stem>.<NN>-<slug>.pvec` with its report beside it. `NN` is the layout's 1-based position among all the drawing's paper layouts,
zero padded to the width of their count and at least two digits; the slug is the name with every byte outside `[A-Za-z0-9._-]` turned into
`-`, runs collapsed, `-` and `.` trimmed, at most 64 bytes, `layout` when nothing is left. The stem is the path's own stem, and `.stem(..)` is
required for bytes. `.sheets(|c| ...)` takes `PvecConversion`'s builder; the profile's space is the layout whatever it sets.

A layout with nothing to print (no entity but its viewports, and no viewport that is on, plotted and has a view, one on a no-plot layer included, #1577) is
`SheetSkip::Empty`; one whose content is all excluded is `EmptyAfterView` (so is a layout holding only its paper view, until the print
profile says which viewport that is); a file an earlier run left at either's path is removed. Any other failure of one sheet is `Failed`,
its path is left as it was, and the rest go on. An index an earlier run left is removed before the first sheet.
The whole set fails only when the destination is not a directory, the stem is missing, the drawing does not open, a selected layout does not
exist, the index cannot be written, or the run is cancelled. Each sheet publishes atomically and the set does not: `<stem>.sheets.json`
(`schema_version` 1: per layout its name, `tab_order`, `nn`, file, status `written`, `empty`, `empty-after-view` or `failed`, the reason,
the entities written and the report's `sheet` section) is written last, so its presence says the run finished. A drawing with no paper
layout writes nothing, not even the index, and never falls back to model space; one whose layouts are all empty writes the index.
