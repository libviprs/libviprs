# ADR: PVec artifacts are reproducible per architecture, not across architectures

Status: Accepted for #1471, found with the drift check of #1475. Part of the integration epic #1039.
Upstream: libviprs/pvec#261.

## Context

The determinism baseline of #1060 showed that 37 of the 39 accepted fixtures of the pinned corpus
(`r2-open`) give the same artifact digest on native x86_64 and native arm64, and that two do not:
`general-affine-chain-001` and `nonuniform-scale-rotated-001`. Two conversions on one host are byte
identical, so it was not run-to-run noise. My first suspect was a trig call in the transform
composition and promotion.

The drift check made it possible to say which number moves. I dumped the geometry that PVec's reader
returns for every converted fixture on both architectures and diffed the dumps. Only those two
fixtures differ, by 1 to 3 ulp:

- `general-affine-chain-001`: the trim span and the y component of the line direction, and through
  them the manifest bounds;
- `nonuniform-scale-rotated-001`: both semi-axes of the ellipse (`6.000000000000001` against `6`,
  `1.9999999999999998` against `2`), and the bounds.

Then I printed what libviprs hands to PVec at each step, on both hosts. Every INSERT's `sin` and `cos`
are identical, every level matrix is identical, and every composed matrix is identical. The inputs of
`Curve::transform` are the same bits, and its outputs are not.

Finally I hashed the results of 200k random inputs per libm function, in `rust:1.99.0-bookworm`
(glibc 2.36) on a native x86_64 host (with FMA) and on native arm64. `sin`, `cos`, `sin_cos`, `tan`,
`exp`, `atan2` and `sqrt(x*x + y*y)` agree bit for bit. `hypot` does not. `hypot` is called by PVec's
`Vector2::length`, `Point2::distance`, `Bounds::diagonal` and by `Policy::normalise_direction` (NP-10),
which are under the line and ellipse images of `Curve::transform`.

So the trig suspicion was wrong, and the cause is `f64::hypot` inside PVec.

## Decision

Artifacts are reproducible per architecture. libviprs does not claim that one drawing gives the same
bytes on x86_64 and arm64, and the checked-in digests say so:

- the x86_64 digest is the base file and arm64 rows that differ live next to it in
  `<name>.aarch64.baseline` (`determinism` for the artifact bytes, `drift` for the geometry the reader
  returns) and in `tests/fixtures/pvec-drift/accepted.aarch64.dump` for the accepted geometry;
- `tests/pvec_1471_reproducibility.rs` pins the set of fixtures that have such a row to the list
  below, so a third divergence cannot be added without this ADR changing;
- the drift check treats the two outputs as the same drawing: a conversion on either architecture is
  clean against the accepted geometry of both, because the difference is far inside `Policy::V1`.

libviprs did not make the arithmetic bit-reproducible, because it cannot do that without copying
PVec's maths. The `hypot` calls are in `pvec-geometry` and `pvec-numeric`, reached through
`Curve::transform`, and `pvec-numeric` has a deterministic `atan` and `atan2` and no `hypot`. Rewriting
the line and conic images in libviprs would be a second copy of the transform rules that the epic's
architecture forbids. The fix belongs upstream: libviprs/pvec#261 asks for `detmath::hypot` (and
`sin_cos`, which `conic.rs` also takes from libm), after which the override rows can be deleted and
this ADR superseded.

What libviprs does own, it makes deterministic. The one `hypot` in libviprs's own PVec code took the
unit vectors of an INSERT array's axes in `flatten.rs` and went into a stored translation. It uses
division and `sqrt` now, which IEEE 754 rounds correctly, and a test scans `src/pvec/` so a libm
`hypot` does not come back.

## Fixtures that differ

- `general-affine-chain-001`
- `nonuniform-scale-rotated-001`

## Consequences

- A fixture that turns a circle into an ellipse or normalises a rotated direction can hit this, so a
  new fixture of that kind will show up as an arm64 row. The drift check says whether it is 1 ulp or a
  real change: a real change is a drift report, a 1 ulp difference is not.
- A digest checked in on one architecture cannot be compared with one from the other, and the gate
  results go on the pull request per architecture.
- Reproducibility across architectures is still the goal. This ADR records where it stops today.

## Alternatives I rejected

- Post-process PVec's output in libviprs to round it to a canonical form. That hides real changes
  below the rounding and still copies PVec's tolerance rules.
- Shadow `hypot` with an exported C symbol. It changes the arithmetic of every other crate in the
  process, and a library must not do that.
- Wait for PVec and say nothing. The two rows were already hiding a question the epic should answer.
