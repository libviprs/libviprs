# ADR: The six classifications and the refusal rule

Status: Frozen by N1.8 (#1047). Decides the policy of N1.3 (#1042) and N1.4 (#1043), part of the
integration epic #1039. The enum itself is N3.5's (#1100) and is not defined here.
Evidence: [docs/spikes/dwg-mapping.md](../spikes/dwg-mapping.md) and
[docs/spikes/view-context.md](../spikes/view-context.md), with the matrix of
[docs/spikes/transform-matrix.md](../spikes/transform-matrix.md) for the degenerate cases.

## Context

Panel finding 5: an entity that cannot be converted must be visible in the output, not absent from
it. Two spikes produced concrete cases (five capability gaps, one refused map, a set of view
states), and this ADR says what each one is called and what is never allowed to happen to it. The
numeric policy is P1.2's, and libviprs defines no epsilon of its own: whether a map is singular or a
conic too eccentric is PVec's answer, and this ADR only names the outcome.

## Decision

Every entity encountered gets exactly one of six classifications, so the six counts sum to the
entities encountered. The meaning column is my summary so the policy reads. N3.5 owns the single enum and its definitions:

| class | meaning | geometry reaches PVec |
|---|---|---|
| `SupportedExact` | PVec holds the same mathematical object, passed through | yes |
| `SupportedEquivalent` | the same locus under a different parameterisation or form (a PVec canonical domain, a Bezier named as a B-spline) | yes |
| `RequiresPVecCapability` | PVec has no public entry point for the construction yet, a gap is filed against `libviprs/pvec` | no |
| `RequiresExtension` | the construct needs a PVec profile extension, not just a constructor | yes, as an extension value |
| `Unsupported` | v1 cannot or will not convert it, with a stable reason code | no |
| `IgnoredByPolicy` | the state exists in the drawing and v1 chooses not to act on it, the report says so | entity treated as if the state were absent |

The refusal rule: an entity that is not `SupportedExact` or `SupportedEquivalent` is counted,
carries a stable reason code, its source class and the rule that decided it, and is never repaired,
approximated or dropped. Silently tessellating an unsupported curve into canonical PVec is a defect
(NG-V1 of [pvec-normalized-geometry.md](pvec-normalized-geometry.md)): the only permitted outcomes for
a curve PVec cannot take are `RequiresPVecCapability` or `Unsupported`. The test of N3.5 that
tessellates a curve and expects a canonical PVec value to fail is the executable form of this rule.
A display-fidelity result is never `SupportedExact`, because exactness is claimed only through the
conformance classes of EPIC P.

Frozen assignments so far, each with the spike that measured it:

- Bulge segments, ellipses and elliptical arcs, periodic B-splines: `RequiresPVecCapability` until
  libviprs/pvec#91, #92, #94 and #93 (closed by #98 after the measured pin) are consumed.
- A singular, ill-conditioned or non-finite insert map: `Unsupported`, codes
  `TRANSFORM_SINGULAR`, `TRANSFORM_CONIC_ILL_CONDITIONED`, `TRANSFORM_NON_FINITE`
  (see [pvec-transform-matrix.md](pvec-transform-matrix.md)).
- Paper space and viewport clips: `Unsupported`, `VIEW_NOT_ENABLED` (508).
- An xref or an INSERT of no block: `Unsupported`, `UNRESOLVED_BLOCK` (105).
- An entity on an off or frozen layer, or under such an INSERT: excluded and counted per layer
  (`LAYER_HIDDEN`, proposed). A viewport-only freeze: `IgnoredByPolicy` (`VIEWPORT_STATE_IGNORED`,
  proposed). N4.6 allocates the two proposed codes.
- z, layer state and a dropped duplicate closing vertex have no PVec home. The provider counts and
  reports each by name.

## Numbers

N1.3 over 30 fixtures at PVec `d399d76`: 16 exact, 8 equivalent, 5 `RequiresPVecCapability`, 1 refused,
0 failed. N1.5 over 127 records: every non-degenerate cell had an exact PVec answer, every
degenerate one was refused with the named rule and no curve emitted. N1.4 on `vc_layers`: the decoder
emitted 13 records where the rule selects 3, which is the count that makes `LAYER_HIDDEN` necessary.

## Consequences

- N3.5 implements the six classes and the reason-code list in one place, and a mapping may not
  introduce a seventh.
- Gaps close by consuming PVec's constructors, never by a local workaround.

## Open items

- Whether a difference that is only PVec's canonical parameter domain (a trimmed line over its
  length, a trimmed arc over its span) is `SupportedEquivalent` or `SupportedExact` is N3.5's call.
  N1.3 graded it equivalent under the strict reading and I have not decided it. Not measured against
  the conversion report's noise.
- The reason-code names marked proposed do not exist in `DiagnosticCode` yet, and N4.6 (#1077)
  allocates them.
- The gap rows moved after the measured pin (#98 closed #93). Re-measuring them against the newer PVec
  is unmeasured, and is phase 4's first step for each class.
- The grades rest on `READER_MAPPING_ONLY` fixtures (no vendor writer, no STEP oracle).
