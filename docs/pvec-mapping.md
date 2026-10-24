# Normalized geometry to PVec (N6.2, issue #1089)

This file is generated from `NORMALIZED_TO_PVEC` in `src/pvec/mapping.rs`. Do not edit it by hand: a test renders the table and fails if the committed file differs by a byte. To change it, change the table and run `PVEC_MAPPING_BLESS=1 cargo test --features pvec-normalized --test pvec_1089_mapping`.

The capability ids are PVec's registry ids. The last column is what a record is classified as when PVec does not have that capability. Tessellation is never an answer, and no curve row maps to a polyline or a point set.

| normalized | PVec type | conformance class | capability | when absent |
|---|---|---|---|---|
| `Line` | `trimmed_curve(line)` | structural exactness | 8 `trimmed_curve` | `RequiresPVecCapability` |
| `Circle` | `circle` | structural exactness | 6 `circle` | `RequiresPVecCapability` |
| `CircularArc` | `trimmed_curve(circle)` | structural exactness | 8 `trimmed_curve` | `RequiresPVecCapability` |
| `Ellipse` | `ellipse` | structural exactness | 7 `ellipse` | `RequiresPVecCapability` |
| `EllipticalArc` | `trimmed_curve(ellipse)` | structural exactness | 8 `trimmed_curve` | `RequiresPVecCapability` |
| `Polyline` | `polyline` | structural exactness | 5 `polyline` | `RequiresPVecCapability` |
| `Composite` | `composite_curve` | geometric exactness | 9 `composite_curve` | `RequiresPVecCapability` |
| `Bezier` | `bezier_curve` | structural exactness | 10 `bezier_curve` | `RequiresPVecCapability` |
| `BSpline` | `b_spline_curve` | structural exactness | 11 `b_spline_curve` | `RequiresPVecCapability` |
| `RationalBSpline` | `rational_b_spline_curve` | structural exactness | 12 `rational_b_spline_curve` | `RequiresPVecCapability` |
