# Transform promotion and degeneracy matrix (N1.5)

Spike for #1044, part of the integration epic #1039. This is the table that `N3.4` composes against
and that `N1.8`, `N2.6` and `N4.4` read, so it has to exist before a single matrix gets multiplied.

An insert's affine transform can turn a circle into an ellipse, a curve into its mirror image, or a
whole branch into nothing. I did not want any of that decided at a call site, so this is one matrix
with a measurement behind every cell. Rows are the source shapes of the N1.2 mapping table
(`docs/adr/pvec-normalized-geometry.md`, NG-03), plus two I added because the ADR names them in
passing (a bulge polyline is a composite, an elliptical arc is a trimmed ellipse). Columns are the
seven transform classes of the issue.

## What I measured, and against what

I pushed nine fixtures through one instance of each transform class using PVec's own transform entry
point (`Curve2d::transform` in `pvec-geometry`, P3.8, libviprs/pvec#49 closed by libviprs/pvec#81),
at libviprs/pvec revision `1f7561e`. Every promotion target below is what that call returned, and
every comparison goes through `Policy::V1` of `pvec-numeric`: PVec numeric policy v1, NP-04 for the
tolerance and NP-05 for equality. The numbers come from `transform-matrix/results-x64.tsv`, written
by `transform-matrix/run.sh` on native x86_64 (MARS, rustc 1.99.1). Nothing in the harness defines
a tolerance, evaluates a curve or flattens one: it carries no numeric literal of its own for either,
and a test fails if it grows one.

The promotion rules themselves are PVec's, not mine: `docs/ISO_PROFILE.md` section 13 (the promotion
table) and ADR 0012 in libviprs/pvec. This spike confirms them by measurement from the libviprs side
and records the outcome each cell would have in a conversion report. I invented no target and no number.

How a measured point is judged: for each fixture I sample 256 parameters plus both ends, map the
original point with the transform, evaluate the transformed curve at the mapped parameter, and take
the largest distance. That worst deviation is compared with `tol(scale, mag)` of NP-04, where the
scale is the extent of the mapped sample set and the magnitude is its largest coordinate. The ratio
column of the Evidence table is deviation over tolerance, so anything at or under one passes.

Fixtures (parameters, not drawings; the probes of O1.6 in libviprs/pvec-gold-fixtures are still open):

| id | row | definition |
|---|---|---|
| F1 | line | a trimmed line, origin (1, 2), direction (3, 4), trim 0.5 to 6 |
| F2 | polyline | four vertices (0,0) (4,0) (5,3) (2,5) |
| F3 | polyline with bulge | composite: a trimmed line (0,0) to (4,0), then a quarter arc of the circle at (4,2), radius 2 |
| F4 | circular arc | circle centre (3,-2), radius 5, reference direction (1,0), trim 0.3 to 2.1, sense true |
| F5 | circle | the circle of F4, untrimmed |
| F6 | ellipse | centre (1,1), reference direction (0.8,0.6), semi-axes 4 and 2 |
| F7 | elliptical arc | the ellipse of F6, trim 0.4 to 4.0, sense true |
| F8 | b-spline | degree 3, six control points, clamped knots 0,0,0,0,1,2,3,3,3,3 |
| F9 | nurbs | degree 2, five control points, knots 0,0,0,1,2,3,3,3, weights 1, 0.7, 1.4, 1, 0.9 |

Transform instances (one per class, plus the cases the issue singles out):

| id | class | map |
|---|---|---|
| T1 | translation | translate(7.5, -3.25) |
| T2 | rotation | rotate(0.6 rad) |
| T3 | uniform scale | scale(2.5, 2.5) |
| T4 | non-uniform scale | scale(3, 0.5) |
| T5 | reflection | mirror_x, determinant negative |
| T6 | shear / general affine | [1.3 0.4 -0.7 0.9 | 2 -1], shear and anisotropy together |
| T7 | singular | [1 2 2 4 | 0 0], determinant exactly zero |
| T8 | pure shear | shear(0.7, 0), used for the ellipse-under-shear case |
| T9 | near-singular, below | scale(1, 0.5 x the policy REL), refused side of the NP-19 bound |
| T10 | near-singular, above | scale(1, 2 x the policy REL), accepted side of the NP-19 bound |
| T11 | non-finite coefficient | a NaN in the first entry |

## The matrix

Each cell is the exact target PVec semantic type after the transform, or an N3.5 classification with
its stable reason code, followed by the evidence id of the measurement behind it. The type names are
the ISO 10303-42 construct names of the PVec profile, as in NG-03.

| Row | Translation | Rotation | Uniform scale | Non-uniform scale | Reflection | Shear / general affine | Singular |
|---|---|---|---|---|---|---|---|
| line | `trimmed_curve over line` E1 | `trimmed_curve over line` E10 | `trimmed_curve over line` E19 | `trimmed_curve over line` E28 | `trimmed_curve over line` E37 | `trimmed_curve over line` E46 | `Unsupported` `TRANSFORM_SINGULAR` E55 |
| polyline | `polyline` E2 | `polyline` E11 | `polyline` E20 | `polyline` E29 | `polyline` E38 | `polyline` E47 | `Unsupported` `TRANSFORM_SINGULAR` E56 |
| polyline with bulge | `composite_curve` E3 | `composite_curve` E12 | `composite_curve` E21 | `composite_curve` E30 | `composite_curve` E39 | `composite_curve` E48 | `Unsupported` `TRANSFORM_SINGULAR` E57 |
| circular arc | `trimmed_curve over circle` E4 | `trimmed_curve over circle` E13 | `trimmed_curve over circle` E22 | `trimmed_curve over ellipse` E31 | `trimmed_curve over circle` E40 | `trimmed_curve over ellipse` E49 | `Unsupported` `TRANSFORM_SINGULAR` E58 |
| circle | `circle` E5 | `circle` E14 | `circle` E23 | `ellipse` E32 | `circle` E41 | `ellipse` E50 | `Unsupported` `TRANSFORM_SINGULAR` E59 |
| ellipse | `ellipse` E6 | `ellipse` E15 | `ellipse` E24 | `ellipse` E33 | `ellipse` E42 | `ellipse` E51 | `Unsupported` `TRANSFORM_SINGULAR` E60 |
| elliptical arc | `trimmed_curve over ellipse` E7 | `trimmed_curve over ellipse` E16 | `trimmed_curve over ellipse` E25 | `trimmed_curve over ellipse` E34 | `trimmed_curve over ellipse` E43 | `trimmed_curve over ellipse` E52 | `Unsupported` `TRANSFORM_SINGULAR` E61 |
| b-spline | `b_spline_curve_with_knots` E8 | `b_spline_curve_with_knots` E17 | `b_spline_curve_with_knots` E26 | `b_spline_curve_with_knots` E35 | `b_spline_curve_with_knots` E44 | `b_spline_curve_with_knots` E53 | `Unsupported` `TRANSFORM_SINGULAR` E62 |
| nurbs | `rational_b_spline_curve` E9 | `rational_b_spline_curve` E18 | `rational_b_spline_curve` E27 | `rational_b_spline_curve` E36 | `rational_b_spline_curve` E45 | `rational_b_spline_curve` E54 | `Unsupported` `TRANSFORM_SINGULAR` E63 |

How to read it:

- A cell that names a PVec type is a `SupportedExact` result: PVec returns the exact image under its
  promotion table and libviprs does nothing but carry it. No cell needed `SupportedEquivalent`,
  `RequiresPVecCapability` or `RequiresExtension`, because PVec has an exact answer for every kind
  under every non-degenerate class. The exact rational form of a conic is a conversion a consumer can
  ask for (`Curve::to_rational_b_spline`, E100 to E107, which is the `SupportedEquivalent` entry
  point), and it is never what a transform returns.
- Whether a circle stays a circle is PVec's decision (a similarity keeps it, anything else does not),
  made under its own equality rule. libviprs does not classify the map as "uniform" or "non-uniform"
  itself, because that would be a second copy of a rule. `N3.4` should call the transform and check
  the returned kind against this table. That is a recommendation for `N1.8`, flagged under the open items.
- The singular column is the same for every row: the map cannot be built, so no curve exists to
  promote. See "Singular and near-singular transforms".
- One exit applies to every cell whose result is a conic, and it is not a column: an image too
  eccentric for NP-07. See "Conditioning exit".

## Findings the cells rest on

### Circle under non-uniform scale

An exact ellipse, no capability gap. Circle F5 (radius 5) under T4 comes back as an `ellipse` with
semi-axes (15.000000000,2.500000000) and the reference direction (1.000000000,0.000000000), worst deviation
3.553e-15 (1.168e-6 of the NP-04 tolerance), evidence E32. The same circle under
the general affine T6 is an ellipse again (E50, parameter shifted by 0.089926750), and under pure shear
too (E68). The arc of F4 under T4 stays a trimmed curve whose basis is promoted, `trimmed_curve over
ellipse` (E31), with its end points preserved. A converter that returns a circle here is wrong, not
different, which is why the cell is pinned.

### Arc under reflection

The arc F4 (centre (3, -2), radius 5, trims 0.3 to 2.1, sense true) under T5 (mirror across the x axis,
determinant negative) is measured in E40. The result is still `trimmed_curve over circle`; what changes
is orientation and parameter direction, and both are reproducible from the record:

- Sweep: the traversal turns counter-clockwise before and clockwise after:
  sweep_before=ccw, sweep_after=cw.
- Endpoint order: the arc keeps its end points and their order. The start maps to the start and the
  end to the end (start_preserved=true, end_preserved=true), and the
  trimmed parameter is not rescaled.
- Reversed flag: the stored canonical form flips from reversed_before=false to
  reversed_after=true.
- Basis: the new circle is centred at (3.000000000,2.000000000), radius 5.000000000, and its frame is
  re-expressed right-handed, reference direction (1.000000000,0.000000000).

To redo it by hand with only the profile's rules: the basis parameter map is `t' = t0 - t` because the
determinant is negative, where `t0` is the profile's shift for the image axes (zero here, since the images of
the two axes are already orthogonal). The trim ends map to {-0.3, -2.1} with the sense inverted by the
reflection, so sense false, and NP-11 normalises a false-sense pair to the increasing interval starting
at the smaller end plus the reversed flag. That gives trim start 4.183185307 (which is 2 pi minus 2.1)
and span 1.800000000, matching the measurement. The ellipse arc F7 behaves the same way (E43).

### Ellipse under shear

An ellipse, not a rational B-spline. PVec's ADR 0012 deliberately differs from the first plan here:
the affine image of an ellipse is an ellipse, so it is exact and simpler. Ellipse F6 under pure shear T8
is an `ellipse` with semi-axes (5.477959302,1.460397852), reference direction (0.882064907,0.471127902),
worst deviation 2.701e-15 (2.382e-6 of the tolerance), evidence E69; under the
general affine T6 it is E51. A consumer that wants a spline asks for the exact rational form, which is
the `SupportedEquivalent` entry point `Curve::to_rational_b_spline` and which succeeded on the original
and on the sheared ellipse and arc (E100 to E107, nine control points for a full conic). A polyline fallback
exists nowhere on this path.

### NURBS under a general affine

NURBS F9 under T6 stays a `rational_b_spline_curve`, evidence E54. The control points are the mapped
control points: compared one by one the largest difference is 0.000e0. The weights are
preserved (weights_identical=true), the degree is unchanged and the knot vector is not touched
at all (knots_identical=true). The pointwise check of the curve itself agrees to
3.794e-15 (3.166e-6 of the tolerance). The same holds for the plain B-spline F8 (E53) and under
every other class. The parameter map of a spline is the identity.

### Singular and near-singular transforms

A singular transform is refused, never approximated. `Transform2::new` refuses the map and PVec's
refusal names the rule: PVec numeric policy v1, NP-19. Because a refused map cannot be built, no curve
can reach it, so no zero-area curve and no best-effort output exists to be returned. E55 to E63 show the
exactly singular map T7 refused for every fixture with no curve emitted. This is a refusal, not a
best-effort conversion.

What the conversion report says. Every entity under the insert is counted in `Unsupported`, not
dropped, so the six counts still sum to the entities encountered (N3.5), and each carries the reason code
`TRANSFORM_SINGULAR`, its source class, and the rule it came from (PVec numeric policy v1, NP-19). The
branch is reported once per entity and nothing under it reaches PVec. N3.4 attaches the insert whose
transform it was, as it does for a non-finite coefficient.

Near-singular handling is the same call and the same rule. PVec numeric policy v1, NP-19 states the
condition under which a map is singular or near-singular, and `Transform2::new` applies it, so this spike
and libviprs define no threshold of their own and nothing here restates one. To see which side of the
bound a map falls on, the harness derives its two maps from the policy's own constant by name, one on
each side, and lets PVec decide. The map on the refused side is refused with NP-19 for every fixture
(E73 to E81). The map on the accepted side is accepted by NP-19 (E82 to E90), which is where the second
conditioning rule matters, see below.

### Conditioning exit

A map can pass NP-19 and still produce a conic whose semi-axes are too far apart. PVec numeric policy
v1, NP-07 refuses that image, and nothing falls back to anything else. Measured on the accepted-side map,
the circle, the ellipse, both arcs and the bulge composite (which contains an arc) are refused with NP-07
(E84 to E88), while the line, the polyline, the B-spline and the NURBS are accepted (E82, E83, E89, E90),
since a hull that goes linear is a legal curve under NP-08. A conic cell in the matrix is therefore
exact for every map NP-07 allows and `Unsupported` with the code `TRANSFORM_CONIC_ILL_CONDITIONED` for
the rest. I did not move the line between the two, PVec does.

## Reflection rows

Orientation here means the turning direction of the traversal in source parameter order, as measured by
the sign of the cross product of the first and second derivative at the middle of the domain. Parameter
direction is how the transformed curve's own parameter relates to the source parameter (`t_new = offset + scale * t`,
the `ParamMap` of PVec).

| row | orientation | parameter direction | evidence |
|---|---|---|---|
| line | the direction is reflected, nothing turns | unchanged, the trim keeps its numbers (the length of the image of the direction is one), reversed flag stays false | E37 |
| polyline | vertex order kept, so a closed ring's winding flips | unchanged, vertex i stays at parameter i | E38 |
| polyline with bulge | segment order and joints kept, every bulge arc turns the other way | unchanged, segment k stays in [k, k+1] | E39 |
| circular arc | sweep flips ccw to cw, end points and their order kept | the basis parameter reverses (`t' = t0 - t`), the trimmed parameter does not, reversed flag toggles | E40 |
| circle | the locus is mirrored, the new frame is right-handed | reverses, scale minus one | E41 |
| ellipse | as the circle, axes re-expressed orthogonal and major first | reverses, scale minus one | E42 |
| elliptical arc | sweep flips, end points and their order kept | basis reverses, trimmed parameter does not, reversed flag toggles | E43 |
| b-spline | control points reflected in the same order, so the turning flips | unchanged, knots untouched | E44 |
| nurbs | as the b-spline, weights unchanged | unchanged, knots and weights untouched | E45 |

## Diagnostics

Reason codes this spike proposes for `N3.5`'s single enumeration, each with the PVec rule it is
measured against. `N1.8` freezes the names.

| code | N3.5 class | PVec rule | when |
|---|---|---|---|
| `TRANSFORM_SINGULAR` | `Unsupported` | NP-19 | the composed map is singular or near-singular (E55, E73) |
| `TRANSFORM_CONIC_ILL_CONDITIONED` | `Unsupported` | NP-07 | the map is usable but the image of a circle or ellipse is too eccentric (E84) |
| `TRANSFORM_NON_FINITE` | `Unsupported` | NP-01 | a coefficient of the map is not a finite number (E91) |

## Handed on

- `N3.4`: look the outcome up in the matrix and pin each cell with a red test, as N3.4's own criteria
  ask. The decision of which kind comes back stays with PVec's call.
- `N1.8`: freeze the three codes above, and decide whether libviprs pre-classifies the map class or
  checks PVec's returned kind against the table (my recommendation is the latter).
- `N2.6` and `N4.4`: the singular row is a refusal of the whole branch with every entity counted.
- The N1.2 ADR pins PVec at `e64b1a4`, which predates P3.8. This spike measured `1f7561e`, so the
  citations in NG-03 and NG-09 need re-checking when the pin moves, as that ADR already says.

## What I could not verify

- The reference is PVec's own evaluator applied before the transform. For the conics, lines and
  polylines that is a real check of the transform, but for the splines the check of the transform is the
  exact control-point comparison and the pointwise one is only self-consistency. An independent
  evaluator belongs to the geometry oracle and to the probes of O1.6 (libviprs/pvec-gold-fixtures#7), which
  are not available yet.
- Only native x86_64 was run. I did not run arm64, so a last-bit difference between the two is not ruled out.
- Provider-side transform extraction from a real DWG insert (N4.4) is not exercised, because the fixtures are
  built from parameters.
- The rational form was checked for success and size here. Its locus deviation was measured by PVec in
  libviprs/pvec#81, and I did not repeat that measurement.

## Evidence

Worst observed deviation under the NP-04 tolerance for every measured record. The deviation is in
coordinate units, the ratio is deviation over tolerance, and a dash is a refusal (no curve, so nothing
to measure) or the rational-form conversion. The largest deviation of any accepted case is 5.443e-15
(E33), and the largest ratio is 6.328e-6 (E67).

| id | fixture | transform | worst deviation | ratio to NP-04 tolerance |
|---|---|---|---|---|
| E1 | F1 | T1 | 1.831e-15 | 3.329e-6 |
| E2 | F2 | T1 | 0.000e0 | 0.000e0 |
| E3 | F3 | T1 | 1.776e-15 | 2.809e-6 |
| E4 | F4 | T1 | 1.776e-15 | 2.191e-6 |
| E5 | F5 | T1 | 1.776e-15 | 1.256e-6 |
| E6 | F6 | T1 | 1.986e-15 | 2.221e-6 |
| E7 | F7 | T1 | 1.986e-15 | 2.407e-6 |
| E8 | F8 | T1 | 0.000e0 | 0.000e0 |
| E9 | F9 | T1 | 3.649e-15 | 4.271e-6 |
| E10 | F1 | T2 | 1.917e-15 | 3.486e-6 |
| E11 | F2 | T2 | 1.256e-15 | 1.812e-6 |
| E12 | F3 | T2 | 1.404e-15 | 2.147e-6 |
| E13 | F4 | T2 | 1.986e-15 | 2.437e-6 |
| E14 | F5 | T2 | 2.220e-15 | 1.570e-6 |
| E15 | F6 | T2 | 1.256e-15 | 1.404e-6 |
| E16 | F7 | T2 | 1.110e-15 | 1.369e-6 |
| E17 | F8 | T2 | 1.256e-15 | 1.665e-6 |
| E18 | F9 | T2 | 2.701e-15 | 3.350e-6 |
| E19 | F1 | T3 | 2.512e-15 | 1.827e-6 |
| E20 | F2 | T3 | 0.000e0 | 0.000e0 |
| E21 | F3 | T3 | 1.986e-15 | 1.256e-6 |
| E22 | F4 | T3 | 3.662e-15 | 1.807e-6 |
| E23 | F5 | T3 | 3.972e-15 | 1.123e-6 |
| E24 | F6 | T3 | 3.662e-15 | 1.638e-6 |
| E25 | F7 | T3 | 3.202e-15 | 1.553e-6 |
| E26 | F8 | T3 | 0.000e0 | 0.000e0 |
| E27 | F9 | T3 | 3.972e-15 | 1.860e-6 |
| E28 | F1 | T4 | 3.580e-15 | 3.530e-6 |
| E29 | F2 | T4 | 0.000e0 | 0.000e0 |
| E30 | F3 | T4 | 3.553e-15 | 1.971e-6 |
| E31 | F4 | T4 | 3.553e-15 | 1.617e-6 |
| E32 | F5 | T4 | 3.553e-15 | 1.168e-6 |
| E33 | F6 | T4 | 5.443e-15 | 2.629e-6 |
| E34 | F7 | T4 | 4.052e-15 | 2.260e-6 |
| E35 | F8 | T4 | 0.000e0 | 0.000e0 |
| E36 | F9 | T4 | 3.553e-15 | 1.477e-6 |
| E37 | F1 | T5 | 1.776e-15 | 3.230e-6 |
| E38 | F2 | T5 | 0.000e0 | 0.000e0 |
| E39 | F3 | T5 | 1.986e-15 | 3.140e-6 |
| E40 | F4 | T5 | 3.580e-15 | 4.417e-6 |
| E41 | F5 | T5 | 4.094e-15 | 2.895e-6 |
| E42 | F6 | T5 | 3.202e-15 | 3.580e-6 |
| E43 | F7 | T5 | 4.003e-15 | 4.852e-6 |
| E44 | F8 | T5 | 0.000e0 | 0.000e0 |
| E45 | F9 | T5 | 0.000e0 | 0.000e0 |
| E46 | F1 | T6 | 3.794e-15 | 6.051e-6 |
| E47 | F2 | T6 | 1.831e-15 | 1.891e-6 |
| E48 | F3 | T6 | 3.553e-15 | 3.860e-6 |
| E49 | F4 | T6 | 3.662e-15 | 3.230e-6 |
| E50 | F5 | T6 | 5.329e-15 | 3.003e-6 |
| E51 | F6 | T6 | 3.560e-15 | 3.133e-6 |
| E52 | F7 | T6 | 2.651e-15 | 2.563e-6 |
| E53 | F8 | T6 | 3.662e-15 | 3.310e-6 |
| E54 | F9 | T6 | 3.794e-15 | 3.166e-6 |
| E55 | F1 | T7 | - | - |
| E56 | F2 | T7 | - | - |
| E57 | F3 | T7 | - | - |
| E58 | F4 | T7 | - | - |
| E59 | F5 | T7 | - | - |
| E60 | F6 | T7 | - | - |
| E61 | F7 | T7 | - | - |
| E62 | F8 | T7 | - | - |
| E63 | F9 | T7 | - | - |
| E64 | F1 | T8 | 2.512e-15 | 3.241e-6 |
| E65 | F2 | T8 | 8.882e-16 | 1.023e-6 |
| E66 | F3 | T8 | 2.665e-15 | 3.476e-6 |
| E67 | F4 | T8 | 4.190e-15 | 6.328e-6 |
| E68 | F5 | T8 | 4.620e-15 | 2.928e-6 |
| E69 | F6 | T8 | 2.701e-15 | 2.382e-6 |
| E70 | F7 | T8 | 1.986e-15 | 1.809e-6 |
| E71 | F8 | T8 | 1.776e-15 | 2.018e-6 |
| E72 | F9 | T8 | 3.553e-15 | 3.861e-6 |
| E73 | F1 | T9 | - | - |
| E74 | F2 | T9 | - | - |
| E75 | F3 | T9 | - | - |
| E76 | F4 | T9 | - | - |
| E77 | F5 | T9 | - | - |
| E78 | F6 | T9 | - | - |
| E79 | F7 | T9 | - | - |
| E80 | F8 | T9 | - | - |
| E81 | F9 | T9 | - | - |
| E82 | F1 | T10 | 4.441e-16 | 1.346e-6 |
| E83 | F2 | T10 | 2.068e-25 | 4.142e-16 |
| E84 | F3 | T10 | - | - |
| E85 | F4 | T10 | - | - |
| E86 | F5 | T10 | - | - |
| E87 | F6 | T10 | - | - |
| E88 | F7 | T10 | - | - |
| E89 | F8 | T10 | 1.034e-25 | 1.477e-16 |
| E90 | F9 | T10 | 2.068e-25 | 2.585e-16 |
| E91 | F1 | T11 | - | - |
| E92 | F2 | T11 | - | - |
| E93 | F3 | T11 | - | - |
| E94 | F4 | T11 | - | - |
| E95 | F5 | T11 | - | - |
| E96 | F6 | T11 | - | - |
| E97 | F7 | T11 | - | - |
| E98 | F8 | T11 | - | - |
| E99 | F9 | T11 | - | - |
| E100 | F4 | rational-form | - | - |
| E101 | F4 | rational-form | - | - |
| E102 | F5 | rational-form | - | - |
| E103 | F5 | rational-form | - | - |
| E104 | F6 | rational-form | - | - |
| E105 | F6 | rational-form | - | - |
| E106 | F7 | rational-form | - | - |
| E107 | F7 | rational-form | - | - |
