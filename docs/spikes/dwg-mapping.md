# DWG entity mapping (N1.3)

Spike for #1042, part of the integration epic #1039. No DWG reaches libviprs today, so the claim that
every CAD curve class this epic cares about has an exact PVec target was a claim and nothing more. I
wanted it measured class by class, with the numbers next to it, before phase 4 writes a line of
extraction code... which is the order I want these things in. The deliverable is evidence that source entities map into PVec's public semantic API
(panel finding 1), and not a geometry implementation: where a mapping needed mathematics, I stopped and
recorded the missing PVec capability instead of doing the maths here.

## What I measured, and against what

The corpus is `libviprs/pvec-gold-fixtures` at tag `corpus-r0-open` (commit `f94f7f2`), revision `r0-open`
of ADR 0008 in that repository: 30 probe DWGs written by acadrust (AC1032) and read back by ACadSharp,
each with an `expected.math.json` that was authored before any DWG existed. Read what that buys before
trusting a row. The manifest says `evidence_scope: READER_MAPPING_ONLY` and `oracles: none`. It means two
open readers decode these files to the authored numbers, exactly where the DWG stores a number verbatim and
within PVec numeric policy v1 where it stores a derived one. There is no STEP file, no vendor writer and no
converter in that revision. The two readers share a DWG specification, so a shared misreading of it is
invisible here (the ADR calls the cross-read common-mode limited, and so do I). Everything below is a
reader-to-PVec mapping result. It is not a conformance result, and it says nothing about how an
AutoCAD-class writer would store the same entities. The corpus also leaves out shear, general-affine and
singular transforms (no single INSERT form), and has no 3D POLYLINE, polyface, closed old-style POLYLINE or
bulge on a closing segment, so those are not rows (see "What I could not verify").

What went through the mapping is the ACadSharp record of each probe (`read.acadsharp.json`). Its fields
are the fields of the decoder IR of #1029 one for one (centre, radius, angles, major axis and ratio,
vertices and bulges, degree, knots, controls, weights, closed and periodic flags, INSERT point, rotation
and scale), and the harness checks that the other reader's record (`read.acadrust.json`) is identical, field
for field, on every fixture (`readers_identical=true` in all 35 records that read a fixture; the sixth probe, P6, is synthetic input). I did not run the decoder itself,
because the crate under test has none that reads a DWG, so "through the decoder" here means "from the
decoder's own record shape".

The target is PVec's public API (`pvec-geometry` and `pvec-numeric`, libviprs/pvec at `d399d76`, current
`origin/main`), and every curve operation in a mapping is a call into it: `Line::new`, `Placement2::new`,
`Circle::new`, `Ellipse::new`, `Trimmed::new`, `Polyline::new`, `Composite::new`, `BSpline::new`,
`RationalBSpline::new`, `Transform2` and `Curve2d::transform`. Nothing is tessellated, so none of these rows
is tessellated to make it green.

The harness is `dwg-mapping/harness`, started by `dwg-mapping/run.sh`. The numbers are
`dwg-mapping/results-x64.tsv`, from native x86_64 (MARS, rustc 1.99.1), and the same harness on native
arm64 (Apple Silicon, `docker --platform linux/arm64`) is `dwg-mapping/results-arm64.tsv`. The tables and
prose quote the x64 file. All 30 verdicts are the same on both machines, and the numbers agree except one
(nonuniform-scale-rotated-001: 4.441e-16 on x64, 8.882e-16 on arm64, a last-bit difference in the system
sine and cosine). The harness carries no tolerance of its own (a test fails if it grows one): each
comparison is either bit equality, for a number the DWG stores verbatim, or `Policy::V1.near` at the
feature's own scale, for a derived number, and the corpus's own `comparison.json` draws the same line. The
worst deviation of a row is the largest absolute difference over its checks, and the ratio is that
difference over the NP-04 tolerance of the same check (so a ratio under 1 means inside policy). Where the
harness does arithmetic of its own, it is labelled in the source: `truth side` (turn a truth record's
parameters into points to compare against) or `candidate` (compute what a missing PVec constructor would
have to return, to measure the gap). Neither is libviprs code, and NG-B1 and NG-B5 of the N1.2 ADR keep
both out of it.

A check that cannot fail proves nothing, so I broke one on purpose: flipping the sense of the bulge
candidate turned both bulge rows to `FAILED` on the sweep-sign check and on the end-point checks of the candidate
arc, and I put it back.

How a row is graded, in the vocabulary of N3.5 (#1100) and NG-07:

- `exact`: PVec's value of the type NG-03 names holds the source's structural parameters (verbatim bits
  where the DWG stores them, policy-equal where it stores a derived value), and its parameter domain is the
  truth record's own.
- `equivalent`: the same, but the parameter domain differs from the truth record's. PVec's trimmed line
  runs over `[0, length]` and a trimmed arc over `[0, span]` (profile section 8), and the corpus's
  `PARAMETERIZATION_PRESERVED` assertion says "a reparameterised equivalent counts as geometric only". So
  this is a strict reading, and I flag the consequence for N3.5 in "Handed on".
- `RequiresPVecCapability`: no PVec entry point does the conversion without libviprs doing mathematics, so
  the row names the missing capability. The deviation shown is of the candidate computation, which is the
  size of what PVec would have to deliver.
- `refused`: PVec refuses the input and the corpus's truth says a refusal is the right answer.

## Result table

| row | DWG class | fixture | PVec target | result | worst deviation | evidence | ratio to NP-04 |
|---|---|---|---|---|---|---|---|
| M01 | LINE | `line-001` | trimmed_curve over line | equivalent | 4.441e-16 | `line-001` | 1.123e-6 |
| M02 | LWPOLYLINE (open) | `polyline-001` | polyline | exact | 0.000e0 | `polyline-001` | 0.000e0 |
| M03 | LWPOLYLINE (closed) | `lwpolyline-closed-001` | polyline | exact | 0.000e0 | `lwpolyline-closed-001` | 0.000e0 |
| M04 | POLYLINE (2D, open) | `polyline2d-001` | polyline | exact | 0.000e0 | `polyline2d-001` | 0.000e0 |
| M05 | LWPOLYLINE (open) | `large-coords-001` | polyline | exact | 0.000e0 | `large-coords-001` | 0.000e0 |
| M06 | LWPOLYLINE (bulge positive) | `bulge-arc-001` | composite_curve of trimmed_curve over circle (RequiresPVecCapability: bulge-to-arc constructor (chord and bulge to centre, radius and trim)) | RequiresPVecCapability | 8.882e-16 | `bulge-arc-001` | 2.270e-6 |
| M07 | LWPOLYLINE (bulge negative) | `bulge-arc-neg-001` | composite_curve of trimmed_curve over circle (RequiresPVecCapability: bulge-to-arc constructor (chord and bulge to centre, radius and trim)) | RequiresPVecCapability | 4.441e-16 | `bulge-arc-neg-001` | 1.137e-6 |
| M08 | CIRCLE | `circle-001` | circle | exact | 0.000e0 | `circle-001` | 0.000e0 |
| M09 | CIRCLE | `tiny-radius-001` | circle | exact | 0.000e0 | `tiny-radius-001` | 0.000e0 |
| M10 | ARC | `arc-seam-001` | trimmed_curve over circle | equivalent | 0.000e0 | `arc-seam-001` | 0.000e0 |
| M11 | ELLIPSE | `ellipse-001` | trimmed_curve over ellipse (RequiresPVecCapability: ellipse from a major-axis vector and a ratio) | RequiresPVecCapability | 2.220e-16 | `ellipse-001` | 7.401e-7 |
| M12 | ELLIPSE (arc) | `elliptical-arc-001` | trimmed_curve over ellipse (RequiresPVecCapability: ellipse from a major-axis vector and a ratio) | RequiresPVecCapability | 4.441e-16 | `elliptical-arc-001` | 1.480e-6 |
| M13 | SPLINE | `bspline-d3-001` | b_spline_curve_with_knots | exact | 0.000e0 | `bspline-d3-001` | 0.000e0 |
| M14 | SPLINE | `bspline-d5-001` | b_spline_curve_with_knots | exact | 0.000e0 | `bspline-d5-001` | 0.000e0 |
| M15 | SPLINE | `repeated-knot-c0-001` | b_spline_curve_with_knots | exact | 0.000e0 | `repeated-knot-c0-001` | 0.000e0 |
| M16 | SPLINE | `closed-by-endpoint-001` | b_spline_curve_with_knots | exact | 0.000e0 | `closed-by-endpoint-001` | 0.000e0 |
| M17 | SPLINE | `closed-periodic-001` | b_spline_curve_with_knots (RequiresPVecCapability: periodic (wrapped) B-spline, or an exact unwrap to the clamped form) | RequiresPVecCapability | 0.000e0 | `closed-periodic-001` | 0.000e0 |
| M18 | SPLINE (rational) | `nonunit-weight-001` | rational_b_spline_curve | exact | 0.000e0 | `nonunit-weight-001` | 0.000e0 |
| M19 | SPLINE (rational) | `nurbs-d3-001` | rational_b_spline_curve | exact | 0.000e0 | `nurbs-d3-001` | 0.000e0 |
| M20 | SPLINE (rational) | `nurbs-d2-offconic-001` | rational_b_spline_curve | exact | 0.000e0 | `nurbs-d2-offconic-001` | 0.000e0 |
| M21 | SPLINE (single span) | `bezier-d3-001` | b_spline_curve_with_knots | equivalent | 0.000e0 | `bezier-d3-001` | 0.000e0 |
| M22 | INSERT x3 over LINE | `nested-blocks-001` | trimmed_curve over line | equivalent | 8.882e-16 | `nested-blocks-001` | 1.291e-6 |
| M23 | LINE x3 on three layers | `layer-state-001` | trimmed_curve over line | equivalent | 8.882e-16 | `layer-state-001` | 1.589e-6 |
| M24 | INSERT x1 over LINE | `translation-001` | trimmed_curve over line | equivalent | 4.441e-16 | `translation-001` | 1.123e-6 |
| M25 | INSERT x1 over LINE | `rotation-irrational-001` | trimmed_curve over line | equivalent | 4.580e-16 | `rotation-irrational-001` | 1.159e-6 |
| M26 | INSERT x1 over CIRCLE | `uniform-scale-001` | circle | exact | 0.000e0 | `uniform-scale-001` | 0.000e0 |
| M27 | INSERT x1 over CIRCLE | `nonuniform-scale-circle-001` | ellipse | exact | 0.000e0 | `nonuniform-scale-circle-001` | 0.000e0 |
| M28 | INSERT x2 over CIRCLE | `nonuniform-scale-rotated-001` | ellipse | exact | 4.441e-16 | `nonuniform-scale-rotated-001` | 1.110e-6 |
| M29 | INSERT x1 over ARC | `reflection-001` | trimmed_curve over circle | equivalent | 1.332e-15 | `reflection-001` | 8.882e-6 |
| M30 | INSERT x1 over LINE | `near-singular-001` | refused with no curve emitted | refused | 0.000e0 | `near-singular-001` | 0.000e0 |

Counts: 16 exact, 8 equivalent, 5 `RequiresPVecCapability`, 1 refused, 0 failed, over the 30 fixtures. The
six extra probes (P1 to P6, in `results-x64.tsv` with kind `probe`) measure alternatives and are not rows.

## Lines and polylines

LINE maps to a trimmed line (M01). `Line::new` normalises the direction and `Point2::distance` gives the
trim length, so both are PVec's. The start point is stored verbatim (bit equal). The end point is not
stored: PVec derives it from origin, direction and length, and it lands 4.441e-16 from the source, 1.123e-6
of the tolerance. The domain is `[0, 3.952847075]` where the truth has `[0, 1]`, hence `equivalent`. I
measured the obvious alternative, probe P1: a two-vertex polyline keeps both end points bit for bit and the
`[0, 1]` domain, but it is a `polyline`, so the corpus's `CLASS_PRESERVED` claim for a line fails. I kept
the NG-03 row (trimmed line) and the `equivalent` grade.

LWPOLYLINE, old-style 2D POLYLINE and the closed case are `exact` (M02 to M05): every vertex is bit equal and
the domain is `[0, n]`. A closed polyline answers the open item of the N1.2 ADR (does the repeated first
vertex survive the provider boundary as a vertex or as a flag?): it arrives as a flag. The record of
`lwpolyline-closed-001` has four vertices and `closed: true`, and the adapter appends a copy of the first
vertex to get PVec's five (a copy, not arithmetic), which reproduces the truth's domain `[0, 4]`. One
thing the corpus does not cover and I measured with a synthetic input (P6, no truth record): a closed
polyline whose last vertex already repeats the first becomes a zero-length last segment after the closing
rule, and PVec refuses it under NP-06. Whoever writes the provider has to decide to drop the duplicate,
and that is a decision, not a gap. `large-coords-001` (coordinates near 1e7 with offsets of 3e-4) is bit
exact too.

## Bulge

M06 and M07 are the two bulge probes: `bulge-arc-001` (bulge +0.4) and `bulge-arc-neg-001` (bulge -0.3).
PVec has the target (profile section 9: a composite of trimmed lines and trimmed circles) and no entry point
that builds it from a chord and a bulge, so the row is `RequiresPVecCapability` (libviprs/pvec#91).

I checked it against the fixture's own expected arc in `expected.math.json` (centre, radius, start angle,
sweep and sense), not against a second libviprs computation: the candidate in the harness is the thing
being measured and the expected arc is the reference. For the positive bulge the candidate gives centre
(1.831250000, 3.212500000), radius 2.837011246, start 4.415023835 and sweep 1.522025508, counter-clockwise,
and the worst deviation from the expected arc is 8.882e-16 (2.270e-6 of the tolerance). For the negative bulge
the expected arc is centre (4.404166667, 2.645833333), radius 3.906889392, start 3.178928551, sweep
1.165827178 and sense `CW`, and the candidate agrees to 4.441e-16 (1.137e-6 of the tolerance). The sign
is the point of that probe: a negative bulge is a clockwise arc, the centre falls on the right of the chord,
and the truth stores it as a canonical increasing interval running against the source (`parameter_direction`
`REVERSED_AGAINST_SOURCE`). PVec holds exactly that as a trimmed circle with the `reversed` flag set
(`pvec_sense=cw`), and a trimmed circle built from the expected numbers ends on both polyline vertices
within policy, in the right order, so the sweep sign is covered, in both directions, with the end points as
the check. The composite of that one segment has the domain `[0, 1]` of the truth's polyline. One honest
detail: for the positive bulge the record states no sense at all (it states one only when the arc runs
against the source), and I read the silence as counter-clockwise. The end-point check would fail if that
reading were wrong.

## Circles, arcs and ellipses

CIRCLE maps to a circle and is `exact` (M08, and `tiny-radius-001` with radius 1.0e-5, M09): centre and
radius are bit equal, the domain `[0, 2 pi]`, and PVec's bounds are the centre plus and minus the radius.
A DWG circle has no reference direction, so the frame uses the x axis, an adapter convention.

ARC maps to a trimmed circle (M10). `arc-seam-001` crosses the seam: start 5.5 and end angle 0.716814693. NP-11
gives start 5.5 and span 1.5, which are the expected start and sweep, and nine points along the arc,
compared with points from the truth record's own numbers, agree with deviation zero. The domain is
`[0, 1.5]` against the truth's `[5.5, 7.0]`, so the row is `equivalent`.

ELLIPSE (M11, M12). The DWG stores a major-axis vector and a ratio, the truth stores a unit axis and two
semi-axes (derived values, so policy comparison). PVec takes the unit axis and the first semi-axis from its
own `Placement2::new` and `Vector2::length`, but the second semi-axis is a product (ratio times length)
that has to happen somewhere, and NG-01 says not here: libviprs/pvec#92. The values come out right (worst
deviation 2.220e-16, 4.441e-16 for the arc), and PVec's own `ellipse` built from the truth's semi-axes is bit
identical to them (probe P2), so nothing else is missing. The second gap is in the whole ellipse: a DWG
ELLIPSE with start parameter 0 and end parameter 2 pi becomes `trimmed_curve over ellipse` with a full-period
span, and picking the untrimmed `ellipse` needs a full-circle test that NG-10 bans: libviprs/pvec#94. The
elliptical arc with start parameter 0.3 and end 2.5 is a trimmed ellipse with trim start 0.3 and span 2.2 (P3,
deviation zero from the truth), and `equivalent` on the domain.

## Splines

SPLINE maps to `b_spline_curve_with_knots`, or `rational_b_spline_curve` when weights are present and never
demoted (NG-03). Every non-periodic row is `exact`: degree, control points, knots (with their multiplicities)
and weights are bit identical, the domain is the truth's, and the knot vector is passed through whole, as
NG-B3 asks.

Repeated interior knot (M15, `repeated-knot-c0-001`). The cubic has the interior knot 0.5 at multiplicity 3,
which is the degree, so under NP-14 it is a legal full-multiplicity knot: the curve is C0 there and nothing
is refused. The mapping did nothing with it: the knot vector went through unchanged (`knots_identical=true`),
and PVec accepted it. PVec's behaviour at the break is exactly what P1.2 says (NP-16): a two-sided first
derivative at 0.5 is refused, and the one-sided limits are (6, 15) from the left and (6, -15) from the
right, two different tangents, with the point itself (3, 1.5) continuous. A consumer comparing at that knot
uses the one-sided limits, as the fixture's `derivative_exclusions` already say.

Closure (M16, `closed-by-endpoint-001`). The source flag is false and the truth says the curve is closed
because its end control points coincide. PVec never infers the flag (NP-20): the spline goes in undeclared,
and `declare_closed` accepts the claim when it is made (P5). So closure by end point is a fact a consumer
can see in the geometry, and a closed flag in PVec means "the source said so".

Periodic (M17, `closed-periodic-001`). A periodic cubic with 6 control points and 7 knots, flagged periodic
and closed by both readers. `BSpline::new` refuses it with `BadKnots` under NP-14, because PVec wants
control points plus degree plus one knots (10 here), and the profile says there is no wrapped periodic
representation in v1. Turning it into the clamped form needs knot insertion, which NG-B3 forbids in libviprs,
so the row is `RequiresPVecCapability` (libviprs/pvec#93) and the curve is not rewritten. I flag a doubt: 7
knots for 6 points is not the usual layout of an unclamped periodic vector, and with no vendor writer in
the corpus I cannot say what AutoCAD stores. That is the first thing #93 has to settle.

Weights (M18 to M20). `nonunit-weight-001` carries a weight of 50 among weights of 1 (ratio 50, far inside the
NP-15 floor) and `nurbs-d2-offconic-001` has weights that do not describe a conic: both are bit identical and
stay rational. A single-span cubic SPLINE (`bezier-d3-001`, M21) maps to a b-spline and is `equivalent`, because
the truth calls it a Bezier. PVec's `bezier_curve` is a storage form of the same thing (profile section 10) and
agrees with the b-spline at 17 sample points (P4, deviation zero: PVec's evaluator against itself, so this is
self-consistency, since the truth only gives control points). PVec's spline bounds are the control hull, a
conservative box and not the tight extent (observed on every spline row, `bounds_is_control_hull=true`).

## Block nest and layers

M22 is `nested-blocks-001`. The corpus nest is three INSERT levels, a translation by (4, 1), a 90 degree
rotation and a scale of (2, 1, 1), over a LINE, so it contains the two-level nest the issue asks for and one
more. Each INSERT becomes PVec's `scale`, then `rotate`, then `translate`, composed with `then` from the
innermost to the outermost. The composed matrix agrees with the truth's, and the line's end points agree
with the truth's world line, to 8.882e-16 (1.291e-6 of the tolerance), so the order of the multiplication is
right (the three do not commute). The grade is `equivalent` for the line's `[0, length]` domain again.

M24 to M30 are the single-transform probes, which overlap with N1.5 (#1044) and are here because they are in
the corpus: translation, an irrational rotation (4.580e-16), a uniform scale (a circle stays a circle), a
non-uniform scale (the circle becomes an ellipse with semi-axes 6 and 2, exact), the same rotated (major
axis (0.6, 0.8), exact), a reflection and a near-singular scale. The reflection (1.332e-15, the worst deviation
in the whole table, 8.882e-6 of the tolerance) keeps the start point as the start and flips the sense of
the arc from counter-clockwise to clockwise, as the truth says. The near-singular scale of 1e-12 is refused
by `Transform2` under NP-19 and no curve is emitted, which is what the truth records as
`NEAR_SINGULAR_TRANSFORM` with disposition `QUARANTINE`; the names differ, because the reason code is
N3.5's to define. One thing that is lost silently if nobody says it: `translation-001` has an insert
point with z = 2, and PVec is exactly 2D, so the world z is dropped. The harness notes it
(`dropped_insert_z`), and the provider must count it.

M23 is `layer-state-001`: three lines on three layers, SHOWN_L (on, thawed), OFF_L (off) and FROZEN_L (frozen).
All three map as lines (3 of 3 encountered, none dropped), and the decoder's layer table agrees with the
truth's state for each. PVec's layer stream holds layer names only, with no on or frozen flag, so the state
lives on the libviprs side, as the fixture itself says: whether an off or frozen layer is drawn is
policy and not a PVec capability. If the policy is to skip them, the record becomes `IgnoredByPolicy` with no
geometry (NG-07) and still counts. I file no PVec gap for it.

## PVec gaps

Real gaps, each a libviprs/pvec issue (four headings, unassigned, referencing #1042 and not closing it):

| gap | missing capability | rows | issue |
|---|---|---|---|
| G1 | bulge to trimmed circle from a chord and a bulge | M06, M07 | libviprs/pvec#91 |
| G2 | ellipse from a major-axis vector and a ratio | M11, M12 | libviprs/pvec#92 |
| G3 | periodic (wrapped) B-spline, or an exact unwrap to the clamped form | M17 | libviprs/pvec#93 |
| G4 | canonical form of a full-period trim of a circle or an ellipse | M11 | libviprs/pvec#94 |

The five candidates of NG-11, each answered against `d399d76`:

- line segment from two points: not a gap. `Line::new(origin, direction)` normalises the direction and
  `Trimmed::new` takes the length from `Point2::distance`, all PVec's; the segment's end is derived, not
  stored, and lands 4.441e-16 from the source (M01).
- polyline bulge: a gap, G1 (libviprs/pvec#91).
- ellipse from a major-axis vector and a ratio: a gap, G2 (libviprs/pvec#92), and a small one (one product).
- bounds of an arc, an ellipse and a spline: not a gap. `Curve2d::bounds` exists for every kind: closed form
  for lines, arcs, ellipses and polylines (exact for the circle and the polyline in the corpus), and the control
  hull, which encloses but is not tight, for splines. If the spatial index needs tight spline bounds that is a
  P-side performance question, not a missing entry point (NG-05).
- angle convention: not a gap. The DWG's ARC angles, ELLIPSE parameters and INSERT rotation come out of the
  decoder in radians, PVec's trims and `Transform2::rotate` are radians (profile sections 6 to 8), and the
  deviations above include the angles. No conversion exists or is needed on this path.

Two non-gaps I want on the record: PVec is 2D by profile, so z is dropped (above), and layer state has no PVec
home by design.

## Decision

Phase 4 may implement as exact: CIRCLE, LWPOLYLINE and old-style 2D POLYLINE (open, and closed through the
flag with the first vertex copied), and every SPLINE that is not periodic (B-spline, NURBS, repeated interior
knots, any degree the probes cover), each passed through whole with no libviprs arithmetic: bit identical to
the source or within policy, with worst deviations of 0.000e0 on every one of those rows. Phase 4 may
implement as equivalent: LINE, ARC and the INSERT-of-LINE and INSERT-of-ARC chains, which differ from the truth
only by PVec's canonical parameter domain, at worst 1.332e-15 (reflection, ratio 8.882e-6 of the tolerance). The
non-uniform-scale chains are exact. INSERT chains compose through `Transform2` and `Curve2d::transform` and
need no libviprs mathematics. A singular chain is refused by PVec under NP-19 and quarantined.

What is deferred, with the capability named and the issue filed: polyline bulge segments (`bulge-arc-001`,
`bulge-arc-neg-001`, libviprs/pvec#91; the candidate matched the expected arcs to 8.882e-16 and 4.441e-16, so
the target is real and only the constructor is missing), ELLIPSE and elliptical arcs (`ellipse-001`,
`elliptical-arc-001`, libviprs/pvec#92 and #94; 2.220e-16 and 4.441e-16), and periodic SPLINE
(`closed-periodic-001`, libviprs/pvec#93; PVec refuses it under NP-14 today). Until those land they report
`RequiresPVecCapability` and are counted, never repaired and never approximated. All of this is a
reader-to-PVec result on `READER_MAPPING_ONLY` evidence: nothing here says how a vendor writer stores these
entities.

## What I could not verify

- The corpus is acadrust-written and cross-read by ACadSharp only (`READER_MAPPING_ONLY`, `oracles: none`,
  common-mode limited by ADR 0008). No vendor writer, no STEP oracle, so a shared misreading of the DWG
  specification would not show.
- Not covered by the corpus, so not measured: 3D POLYLINE, polyface, closed old-style POLYLINE, a bulge on a
  closing segment of a closed polyline, and shear, general-affine and singular transforms (excluded by the
  corpus: no single INSERT form). Also not covered: an extrusion normal other than +z (every probe has +z, the
  harness only checks that), a non-zero block base point (the records carry none), a non-zero elevation, a SPLINE
  built from fit points, and a z scale other than 1.
- The decoder itself did not run: the input is the reader's record, which mirrors the IR of #1029 field for
  field but is not an instance of it.
- The pointwise checks for arcs and ellipses compare PVec's evaluator with a few lines of trigonometry on the
  truth record's own numbers (labelled `truth side` in the harness). They are not an independent oracle.
- The candidate bulge and ellipse computations are spike code that measures the gap. They are not a proposal for
  libviprs and not a reference implementation for PVec.
- Rows M24 to M30 overlap with N1.5 and do not extend it. The line, arc and nest rows are `equivalent` only
  under the strict reading of the parameter domain described above.
- Two runs of one harness on two machines (x64 and arm64) is all the architecture coverage there is. The
  18 checks of `tests/spike_dwg_mapping.rs` ran by hand in a container, std-only, because the hosted test job stops
  at an earlier failing binary (see the pull request).

## Handed on

- N3.5 (#1100): a difference that is only PVec's canonical parameter domain (a trimmed line over `[0, length]`, a
  trimmed arc over `[0, span]`) makes every LINE and ARC `SupportedEquivalent` under the strict reading here.
  If that is too noisy for the conversion report, N3.5 should decide that a domain difference recorded by
  `ParamMap` is `SupportedExact`. I have not decided it for them.
- N1.8 and N3.1: the answers to the open items of the N1.2 ADR are a closed polyline arrives as a flag (the
  first vertex is appended as a copy), and PVec's bounds exist for every kind and are conservative for splines.
- Phase 4 provider work: nothing in PVec carries z, off or frozen layer state, or a dropped duplicate closing
  vertex, so each of those has to be counted by the provider and reported by name.
