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
`control_points` (deprecated, #1554), `profile`, `allowed_capabilities`, `capability_policy`. `probe` returns the per-class counts without writing, `convert`
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
