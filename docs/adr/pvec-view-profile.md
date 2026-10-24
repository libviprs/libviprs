# ADR: What "visible flattened drawing" means for v1

Status: Frozen by N1.8 (#1047) as a rule. Its implementation is blocked on #1239 (see Open items).
Decides N1.4 (#1043), part of the integration epic #1039.
Evidence: [docs/spikes/view-context.md](../spikes/view-context.md), raw output in
`docs/spikes/view-context/results-x64.txt`.

## Context

A DWG has more than one way to answer "what is drawn". Layers hide things, blocks nest, layouts
carry viewports with their own freezes, and xrefs point at other files. Phase 3 and 4 cannot
implement flattening until one answer is fixed, and phases 2 and 4 cannot test it without one.
The numeric policy is P1.2's (PVec numeric policy v1), and libviprs defines no epsilon of its
own here. Nothing in this ADR is a threshold.

## Decision

v1 draws **model space only**, and the visible flattened drawing is the multiset of leaf entities,
each with its composed transform, that these clauses select (the full rule, with the worked
examples, is "The v1 visible flattened drawing" in the spike):

- **V1-1 Space.** Only the model-space view contributes. A layout (paper space) is `Unsupported`,
  refused with a diagnostic, never merged and never silently skipped. Viewport clips, in both the
  exact and the bounded-only case, are deferred with paper space.
- **V1-2 Effective layer.** A leaf's effective layer is its own, except that a leaf on layer "0"
  inside an INSERT takes the layer of the nearest enclosing INSERT.
- **V1-3 Visibility.** An entity is visible only if its effective layer is on and thawed. An
  INSERT that is not visible takes its whole expansion with it. Locked, no-plot and
  viewport-frozen layers stay visible (the last is reported as `IgnoredByPolicy`).
- **V1-4 Instances.** Each visible INSERT contributes its block's visible leaves once per
  instance, a MINSERT being rows times columns instances, under the composed transform.
- **V1-5 External and missing.** An xref, or an INSERT of no block, contributes nothing and is
  reported as `UNRESOLVED_BLOCK` (105). Resolving it, failing the whole conversion and omitting it
  silently were each rejected in the spike, with the reasons written down there.

Every entity excluded by V1-3 is counted per layer in the report, so "nothing was on that layer"
and "the filter lost it" stay distinguishable. Colour, lineweight and linetype play no part.

## Numbers

All native x86_64 on MARS, JIT run of the decoder sources, ACadSharp 3.7.1, no arm64 run:

- `vc_layers`: the decoder emitted 13 records, the rule selects 3. The decoder applies no layer
  state, which is the gap behind #1239.
- `vc_nested`: a three-deep chain with two rotations and two scales lands where the hand
  computation puts it.
- `sample_AC1032`: the rule counts 146 leaves, the decoder emitted 134. All 12 missing are INSERTs
  inside dimension geometry blocks, skipped with no warning (provider limitation L6).
- `sample_AC1032` layouts: 3 layouts with 2 entities each, all VIEWPORTs, 6 `VIEWPORT` warnings.
- `BLOCKVISIBILITYPARAMETER`: 4 instances of a dynamic block, 20 records, the rule agrees.

## Consequences

- N3.3, N4.4, N4.5 and N4.6 implement against V1-1 to V1-5 and do not reopen them. A change to a
  clause is a new ADR.
- The decoder as it stands cannot apply V1-2 or V1-3: layer state is neither applied nor exposed
  on its boundary. That is a provider limitation, not libviprs policy, and the rule stays frozen
  while the provider catches up (#1239, plus the provider traits of N4.1).
- Paper space is a stated v1 non-goal. Anyone who wants it needs a new ADR and a viewport-clip
  measurement against AutoCAD, which does not exist yet.

## Open items

Frozen means the rule is settled. These are the places where the evidence behind it is thin, each
with the measurement that is missing:

- Nothing here is `KNOWN_TRUTH_EXACT`. The phase-1 fixture set of `libviprs/pvec-gold-fixtures#1`
  has no DWG, so layers, layouts, viewports, xrefs and anonymous blocks were measured on four
  spike-authored fixtures plus AutoCAD-authored samples. A spike-authored fixture proves the file
  says what I meant and nothing about what AutoCAD would draw. N2.1 should replace them.
- An entity on an off or frozen layer is not exercised by any real fixture (`sample_AC1032`
  defines `Layer_Off` and `Layer_Freeze` and uses neither in model space). Unmeasured on real data.
- An entity's own invisible flag (DXF group 60) is not read by the decoder or by the rule probe,
  and no fixture sets it. V1-3 does not claim to cover it. Unmeasured.
- Viewport freezes, twisted or non-rectangular viewports and a real xref were measured only on
  writer-authored files. The twist sign convention is unverified, there being no AutoCAD here.
- Whether the layer and block names reach the rule engine by a wire change, a side table or the
  acadrust provider is open and tracked in #1239.
