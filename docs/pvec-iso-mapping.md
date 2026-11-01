# Normalized geometry to PVec

There is one table, `pvec::mapping::NORMALIZED_TO_PVEC`, and ten rows, one for each kind of normalized geometry. It says
which PVec semantic type the kind becomes, which PVec capability that needs, and what the mapping claims
(`docs/pvec-mapping.md` renders the same table with its reasoning). The test `tests/pvec_docs_tables.rs` builds the rows
below from the table and fails if this page differs.

## The table

<!-- mapping:begin -->
| kind | PVec type | capability | conformance |
|---|---|---|---|
| Line | `trimmed_curve(line)` | trimmed_curve (8) | structural exactness |
| Circle | `circle` | circle (6) | structural exactness |
| CircularArc | `trimmed_curve(circle)` | trimmed_curve (8) | structural exactness |
| Ellipse | `ellipse` | ellipse (7) | structural exactness |
| EllipticalArc | `trimmed_curve(ellipse)` | trimmed_curve (8) | structural exactness |
| Polyline | `polyline` | polyline (5) | structural exactness |
| Composite | `composite_curve` | composite_curve (9) | geometric exactness |
| Bezier | `bezier_curve` | bezier_curve (10) | structural exactness |
| BSpline | `b_spline_curve` | b_spline_curve (11) | structural exactness |
| RationalBSpline | `rational_b_spline_curve` | rational_b_spline_curve (12) | structural exactness |
<!-- mapping:end -->

## How to read it

- **StructuralExactness** means the PVec value is the source construct, parameter for parameter.
  **GeometricExactness** means the same locus under an equivalent construct. **DisplayFidelity** would be a densified display
  form, and no row uses it: nothing is tessellated.
- The capability is PVec's own registry id and name. libviprs declares no capability vocabulary; the conversion API's
  allowed-capability gate looks the id up here (`docs/pvec-conversion-api.md`).
- A kind with no row is not carried. It is counted in the report with a reason code (`docs/pvec-dwg-support.md`), never
  reshaped into one that has a row.
