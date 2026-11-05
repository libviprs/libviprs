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

With the pvec crates at afd2fe6 (`detmath::hypot`, pvec#261) the artifact bytes of
`general-affine-chain-001` agree across architectures, so only `nonuniform-scale-rotated-001` keeps an
arm64 row in `determinism.aarch64.baseline`. Both fixtures still have an arm64 row in the drift digest
and the accepted dump, since the rest of their arithmetic differs in the last bit.

## macOS

The rows above are Linux rows, and so is the gate. The fleet is x86_64 and arm64 Linux, and nothing in it runs
a Mac. Issue #1513 reported the digest tests red on Apple silicon, also on plain `main`, so I measured it
instead of guessing, on `aarch64-apple-darwin` (macOS 26.4.1, rustc 1.99.1) with the pvec crates at afd2fe6.
The same two suites are green on arm64 Linux in a container on the same machine.

- `elliptical-arc-001`: one stored bound (the second of the four) is 1 ulp lower than on Linux, `2.016167142459582`
  against `2.0161671424595826`. I read this as Apple's libm rounding `sin` and `cos` differently from glibc, through `sin_cos` in the conic
  path of pvec (pvec#262 only replaced `hypot`), but I did not isolate the call. This is the only fixture where the artifact bytes differ.
- `nonuniform-scale-rotated-001`: `ay` is 1 ulp from the Linux arm64 value and `ry` is 1 ulp from the
  x86_64 one. The drift digest differs and the artifact bytes match Linux arm64.
- `general-affine-chain-001`: identical to the Linux arm64 row, so the drift digest differs from the x86_64 one
  in `trim.span`, `b.dx` (1 ulp) and `b.dy` (2 ulp).

Every difference is within `Policy::V1`, and the drift check (the report, not the digest) stays green on a Mac.
I did not add a macOS row. A row nothing re-runs goes stale with the next OS update, and the person nearest
re-blesses it, which is how a determinism baseline stops meaning anything. The two bit-exact tests,
`n_1471_the_digest_of_every_converted_fixture_matches_the_committed_one` and
`n2_8_1060_the_digest_of_every_conversion_matches_the_checked_in_baseline`, are Linux only: they are ignored
elsewhere with the reason `bit-exact Linux rows, #1513`. The tolerance checks still run on a Mac.
The real fix is a deterministic `sin` and `cos` in pvec's `detmath`, which would also make macOS agree.

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
