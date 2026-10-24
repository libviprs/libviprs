# ADR: The transform promotion and degeneracy matrix

Status: Frozen by N1.8 (#1047). Decides N1.5 (#1044), part of the integration epic #1039.
Evidence: [docs/spikes/transform-matrix.md](../spikes/transform-matrix.md), with
`transform-matrix/results-x64.tsv` and `results-arm64.tsv` beside it. PVec revision `1f7561e`.

## Context

An insert's affine map can turn a circle into an ellipse, mirror an arc or collapse a branch. I did
not want any of that decided at a call site, so the matrix is one table with a measurement behind
every cell. The promotion rules are PVec's (`ISO_PROFILE.md` section 13 and PVec ADR 0012), and
every comparison in the spike goes through `Policy::V1` of P1.2. libviprs defines no epsilon of
its own, and neither does this ADR.

## Decision

Rows are the source shapes of NG-03, plus the bulge composite and the elliptical arc. The matrix
as frozen:

- Under translation, rotation, uniform scale, reflection, non-uniform scale and general affine, every
  kind comes back as the kind PVec's promotion table names, and libviprs carries it. A line stays a
  trimmed line, a polyline stays a polyline, a B-spline and a NURBS keep their degree, knots and
  weights and only their control points move.
- A circle or a circular arc under non-uniform scale or a general affine becomes an ellipse or a
  trimmed ellipse. An ellipse under shear stays an ellipse, not a rational B-spline.
- A bulge composite stays a composite, and the arc segments inside it promote like any arc, with the
  joints kept.
- A reflection keeps the end points and their order, flips the sweep, toggles the reversed flag
  and reverses the parameter of an untrimmed conic. The frame is re-expressed right-handed.
- A singular map is refused, never approximated. No curve is emitted, every entity under the insert
  is counted as `Unsupported`, and the branch is reported once per entity.
- A map that passes the singularity rule can still image a conic too eccentric for NP-07. That is
  refused the same way. Lines, polylines, B-splines and NURBS are unaffected.

Rule for the call site: libviprs does not pre-classify a map as uniform or non-uniform. N3.4 calls
PVec's transform and checks PVec's returned kind against this matrix, because pre-classifying would
be a second copy of a rule.

Reason codes, frozen (all `Unsupported`):

| code | PVec rule | when |
|---|---|---|
| `TRANSFORM_SINGULAR` | NP-19 | the composed map is singular or near-singular |
| `TRANSFORM_CONIC_ILL_CONDITIONED` | NP-07 | usable map, but a circle or ellipse image is too eccentric |
| `TRANSFORM_NON_FINITE` | NP-01 | a coefficient of the map is not finite |

## Numbers

127 records, 9 fixtures plus a closed ring, 13 transform instances, on native x86_64 (MARS) and
native arm64. Every verdict, kind, rule and parameter map agrees across the two architectures, and
the last digits of the deviation differ in 10 of the 127 records. Every accepted case sits far under
the P1.2 tolerance (the spike gives the per-record ratios, E1 to E127, and I do not repeat them
here). The exactly singular map was refused for all 9 fixtures (E55 to E63), the map just on the
refused side of the NP-19 bound for all 9 (E73 to E81), and the non-finite map for all 9 (E91 to E99).
The conic exit was bracketed at a factor of four on each side of the NP-07 bound: inside it every
fixture was accepted (E108 to E116), outside it the circle, ellipse, both arcs and the bulge composite
were refused with NP-07 and the other four kinds accepted (E117 to E125).

## Consequences

- N3.4 pins every cell with a red test and checks the kind PVec returns. N2.6 and N4.4 treat the
  singular column as a refusal of the whole branch with every entity counted.
- The exact rational form of a conic is a conversion a consumer can request
  (`Curve::to_rational_b_spline`), never what a transform returns.

## Open items

- The names in the matrix are what the harness printed for the Rust variant PVec returned. They are
  not read from PVec's capability registry, whose names differ slightly (the profile calls the
  capability `b_spline_curve`). The lookup from variant to registry name is still to do, and needs
  a PVec checkout at `1f7561e` or later, which I did not have to hand. Not measured.
- The reference for splines is PVec's own evaluator, so the pointwise check there is
  self-consistency and only the control-point comparison tests the transform. An independent
  evaluator belongs to the O1.6 probes (libviprs/pvec-gold-fixtures#7), which are not available.
- The NP-07 line is bracketed only at a factor of four and is PVec's to locate. Not measured closer.
- Provider-side extraction of a transform from a real DWG insert (N4.4) is unmeasured, the
  fixtures being built from parameters. A real PDF `cm` stack exercised only translation, apart from
  one reflection (see the PDF ADR).
- Architectures other than native x86_64 and arm64 are not covered.
