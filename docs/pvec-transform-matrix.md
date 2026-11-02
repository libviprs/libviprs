# What a transform does to an entity

An INSERT's affine map can turn a circle into an ellipse, mirror an arc or collapse a branch. The matrix below is the
one table of what comes back, with a measurement behind each cell (`docs/adr/pvec-transform-matrix.md`). The promotion
rules are PVec's: libviprs does not pre-classify a map as uniform or not, it calls PVec's transform and checks PVec's
answer against this matrix. Every comparison uses `Policy::V1`.

## The matrix

The table is `PromotionMatrix::FROZEN`. A row is the source shape, and the cell names what PVec returns under any
non-singular map.

<!-- matrix:begin -->
| source shape | PVec returns |
|---|---|
| Line | TrimmedLine |
| Polyline | Polyline |
| PolylineWithBulge | Composite |
| CircularArc | TrimmedCircle or TrimmedEllipse |
| Circle | Circle or Ellipse |
| Ellipse | Ellipse |
| EllipticalArc | TrimmedEllipse |
| BSpline | BSpline |
| Nurbs | RationalBSpline |
<!-- matrix:end -->

## The degenerate cases

- **Singular.** A singular map is refused, never approximated. No curve is emitted and every entity under the insert is
  counted `Unsupported` with the reason `TRANSFORM_SINGULAR (NP-19)`.
- **Too eccentric.** A map that passes the singularity rule can still image a conic too eccentric for PVec's rule NP-07. That
  is refused the same way. Lines, polylines, B-splines and NURBS are unaffected.
- **Reflection.** Keeps the end points and their order, flips the sweep and toggles the reversed flag.
- **Identity is not a promise of identical bits.** An identity promotion of a circle or an arc can come back a few units in
  the last place away from the untouched curve's box (9,948 of 200,000 bounds in the benchmark, at most 4 ulps). Far inside
  the policy, and recorded in `docs/pvec-benchmarks.md`.

The cost of each class, and of the refusal, is in `docs/pvec-benchmarks.md`.
