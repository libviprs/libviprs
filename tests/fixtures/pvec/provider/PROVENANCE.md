# Provider fixtures

`vc_layers.dwg` and `vc_nested.dwg` are the two acceptance fixtures of N4.1 (#1072). I copied
them byte for byte from `docs/spikes/view-context/fixtures/` on #1235 (N1.4), where the spike
wrote them with ACadSharp's `DwgWriter` 3.7.1 at AC1032. The sha256 values in `SHA256SUMS` are
the ones that spike recorded, and `tests/pvec_provider_acadrust.rs` checks the files against them.

- `vc_layers.dwg`: layers `VC_ON`, `VC_OFF` (off), `VC_FROZEN` (frozen) and `0` (off), four
  lines in model space, and block `VC_BLK` inserted three times on those layers.
- `vc_nested.dwg`: `VC_INNER` inserted at two scales, plus a three-deep chain
  `VC_OUTER` > `VC_MID` > `VC_INNER` with two rotations and two scales.

## What they prove, and what they do not

Written by ACadSharp and read by acadrust, so this is a **cross-family** check: two independent
implementations agreeing about a file one of them wrote. It is **not known truth**. The
expectations in the tests are hand computed from how the drawings were built (the two worked
examples in `docs/spikes/view-context.md` on #1235). Known truth for the provider is the
AutoCAD-authored corpus, recorded in `docs/spikes/acadrust-gate0.md`.

When #1235 merges these two files exist twice; either copy can then go.
