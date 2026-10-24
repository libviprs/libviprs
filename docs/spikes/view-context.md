# View-context flattening and the v1 visible drawing (N1.4)

Issue #1043, parent epic #1039. This is the written definition of "the visible
drawing" that N1.8, N2.6, N3.3, N4.4, N4.5, N4.6 and N4.9 build on, measured on
named fixtures before any flattening code exists.

The short version: v1 draws **model space only**, applies the drawing's own
layer state (off and frozen layers are excluded, and an INSERT on such a layer
takes its whole expansion with it), composes nested inserts exactly, and reports
xrefs as external dependencies. The decoder as it stands cannot do the layer
part, because it neither applies layer state nor exposes it. That is a provider
limitation, recorded below with the versions I saw it on, and it is the main
thing this spike found.

## What I measured, and what I could not

Everything below was run on MARS (native x86_64, `devshell@192.168.0.100`) in
the pinned `mcr.microsoft.com/dotnet/sdk:10.0.401-noble` and `rust:1.99.1-bookworm`
containers through `tools/nas.sh`. The raw output is `view-context/results-x64.txt`
and `view-context/run.sh` plus `stage.sh` and `results.sh` regenerate it. No
arm64 run was made, so none of the numbers here is an arm64 result.

I measured three things, and the document keeps them apart:

- **ACadSharp 3.7.1 object model** (with libviprs-dep's patches applied), read
  by `probe/Program.cs describe`. This says what the provider *knows*.
- **The libviprs-dep decoder**, `3.7.1-viprs.2` at commit `0d475bf`, whose adapter,
  source and wire sources I compiled into the same probe and ran under the JIT
  (`probe/Program.cs shim`). This says what the decoder *emits*. It is not the
  NativeAOT artifact, but it is the same flattener source file for file, which is
  how libviprs-dep's own `fixturegen` works.
- **acadrust 0.5.5** at `d35691f`, read by `acadrust-probe`. This says what the
  other candidate provider for N4.1 *knows*.

What I could not do:

- The phase-1 fixture set in `libviprs/pvec-gold-fixtures#1` has no DWG in it.
  `main` holds the `expected.math.json` records for the transform and
  nested-block probes (O1.4, O1.6), and nothing for layers, layouts, viewports,
  xrefs, anonymous or dynamic blocks. `source.dwg` for every probe waits on the
  ODA writer from O1.2, which is not done. So nothing here is
  `KNOWN_TRUTH_EXACT`.
- I used what exists: AutoCAD-authored drawings from ACadSharp's samples
  (via `libviprs/acadsharp-rs-tests` at `ded7009`), writer-authored fixtures
  already in this repository, and four fixtures I authored for this spike with
  ACadSharp's `DwgWriter`. A fixture written by the same library family that
  reads it only proves the file says what I meant. It says nothing about what
  AutoCAD would draw, and the table below marks them that way.
- I have no AutoCAD, so no viewport transform was checked against a real
  rendering. The twist sign convention in particular is unverified.
- I did not run the NativeAOT build of the decoder, and I did not run
  `AcadSharpDecoder` through the Rust crate. The decoder results are the JIT run
  of the same sources.

## Fixtures

| id | path | origin | evidence class |
|---|---|---|---|
| `vc_layers` | `docs/spikes/view-context/fixtures/vc_layers.dwg` | spike-authored with ACadSharp `DwgWriter` 3.7.1, AC1032. Layers on, off, frozen and an off layer "0"; one block inserted on each state. | spike-authored, read back by ACadSharp only. Not known truth. |
| `vc_nested` | `docs/spikes/view-context/fixtures/vc_nested.dwg` | spike-authored. One block inserted twice at scale 1 and 3, and a three-deep chain with rotations and a scale. | spike-authored, hand-computed expectation in the worked example. Not known truth. |
| `vc_paper` | `docs/spikes/view-context/fixtures/vc_paper.dwg` | spike-authored. Model space with a layer frozen in one viewport only, and a layout with three viewports (1:1, 1:2 with a 30 degree twist and a viewport freeze, and a circle-clipped one). | spike-authored, read back by ACadSharp and acadrust. Not known truth. |
| `vc_anonymous` | `docs/spikes/view-context/fixtures/vc_anonymous.dwg` | spike-authored. An anonymous block `*U1` and a named block, on named layers. | spike-authored. Not known truth. |
| `g13_xref` | `tests/fixtures/cad/g13/g13_xref.dwg` | libviprs-dep `g13_xref.dwg`, written with ACadSharp `DwgWriter`. One INSERT of an xref block. | writer-authored regression fixture. |
| `sample_AC1018` | `tests/fixtures/cad/sample_AC1018.dwg` | ACadSharp `samples/sample_AC1018.dwg`, AutoCAD-authored. Same drawing as `sample_AC1032`, and the rule numbers are identical on all seven saves. | real-world, AutoCAD-authored. |
| `sample_AC1032` | external: libviprs/acadsharp-rs-tests@ded7009 `fixtures/dwg/sample_AC1032.dwg` | ACadSharp `samples/sample_AC1032.dwg`, AutoCAD-authored. 19 layers including `Layer_Off`, `Layer_Freeze` and `Layer_vp_freeze`; three paper layouts with viewports; anonymous `*D`, `*T` and `*U` blocks; one dynamic block. | real-world, AutoCAD-authored. |
| `sample_AC1014` | external: libviprs/acadsharp-rs-tests@ded7009 `fixtures/dwg/sample_AC1014.dwg` | ACadSharp `samples/sample_AC1014.dwg`, AutoCAD-authored. The same drawing as `sample_AC1032` saved as R14. | real-world, AutoCAD-authored. |
| `sample_AC1015` | external: libviprs/acadsharp-rs-tests@ded7009 `fixtures/dwg/sample_AC1015.dwg` | ACadSharp `samples/sample_AC1015.dwg`, AutoCAD-authored. The same drawing saved as R2000. | real-world, AutoCAD-authored. |
| `BLOCKVISIBILITYPARAMETER` | external: libviprs/acadsharp-rs-tests@ded7009 `fixtures/dwg/BLOCKVISIBILITYPARAMETER.dwg` | ACadSharp sample. A dynamic block with a visibility parameter, inserted four times. | real-world, AutoCAD-authored. |
| `AecObjects` | external: libviprs/acadsharp-rs-tests@ded7009 `fixtures/dwg/AecObjects.dwg` | ACadSharp sample. Carries a paper viewport whose view direction is (-1,-1,1). | real-world, AutoCAD-authored. |

`fixtures/SHA256SUMS` has the hashes of the four spike-authored files. A DWG
header carries timestamps, so regenerating them gives different bytes: the
committed ones are the ones the numbers below were measured on, and
`results.sh` was run against byte-identical copies.

The external ones are MIT (ACadSharp's own samples, not Autodesk files) and are
pinned by sha256 in that repository's `manifest.toml`. `sample_AC1032` is
`0e8faaca949c9429c92240082d9ea4d3524aca5baf7c7815ec439245c321b528`,
`BLOCKVISIBILITYPARAMETER` is
`be1e24ea0cd5194d0c57935b5018123b7cc981331172a1a2ca7cecc2d9a18e4f`.

## Decision table

The four behaviours mean:

- `included`: the entity or view contributes to the flattened drawing.
- `excluded`: the drawing's own state hides it, so AutoCAD would not draw it
  either. It is not part of the visible drawing, and it is counted in the
  report, never dropped without a count.
- `IgnoredByPolicy`: the state exists in the drawing and v1 chooses not to act
  on it. The entity is treated as if the state were absent, and the report says
  the state was there.
- `Unsupported`: v1 cannot do this. The thing is refused with a diagnostic and
  is never silently left out.

Diagnostics named `proposed` do not exist in `DiagnosticCode` yet. N4.6 allocates
them. Every other one exists today and I checked the name against `src/cad/report.rs`.

| case | v1 behaviour | diagnostic | fixture | basis |
|---|---|---|---|---|
| `model-space` | `included` | none | `vc_nested`, `sample_AC1032` | Decoder view 0 is always `ViewKind::Model`. Measured: every line, arc, circle, ellipse, polyline and spline in model space reaches the sink. |
| `paper-space-layout` | `Unsupported` | `TilerError::ViewNotEnabled` today, `VIEW_NOT_ENABLED` (508) once the decode report carries it | `sample_AC1032`, `vc_paper` | Paper space is out of v1, see the Paper space section. The decoder emits no drawing geometry for a layout of a real AutoCAD drawing: all three layouts of `sample_AC1032` hold two entities each and both are VIEWPORTs. |
| `viewport-clip-exact` | `Unsupported` | `VIEW_NOT_ENABLED` (508) | `vc_paper`, `sample_AC1032` | An untwisted plan-view rectangular viewport has an exact clip: the rectangle `Center +/- (Width, Height)/2`, every field exposed by ACadSharp and acadrust. Deferred with paper space. |
| `viewport-clip-bounded` | `Unsupported` | `VIEW_NOT_ENABLED` (508) | `vc_paper`, `AecObjects` | Only a bound exists for a non-rectangular clip (the boundary is an entity reached by handle), a twisted viewport (sign convention unverified) and a non-plan view direction (a projection, not a 2D affine). Deferred with paper space. |
| `layer-off` | `excluded` | `LAYER_HIDDEN` (proposed), one per layer with a count | `vc_layers` | Measured: ACadSharp exposes `Layer.IsOn`, the decoder ignores it (13 records emitted, 3 visible by the rule). No real fixture puts an entity on `Layer_Off`, see Provider limitations. |
| `layer-frozen` | `excluded` | `LAYER_HIDDEN` (proposed), same counter | `vc_layers` | Measured: `LayerFlags.Frozen` is exposed by ACadSharp and acadrust, ignored by the decoder. Same fixture, same gap. |
| `layer-frozen-in-viewport` | `IgnoredByPolicy` | `VIEWPORT_STATE_IGNORED` (proposed), once per drawing that has any | `vc_paper` | A viewport freeze (`Viewport.FrozenLayers`) acts only inside that viewport. In the model-space baseline the layer is visible, so v1 draws it and reports the freeze as ignored. Measured on `vc_paper`: line `4B` is on `VC_VPF`, frozen in viewport `4F` only, and is emitted. No real fixture has a viewport freeze. |
| `insert-layer-override` | `excluded` | `LAYER_HIDDEN` (proposed) | `vc_layers` | The only per-insert layer effect the object model carries is the INSERT's own layer: an INSERT on an off or frozen layer takes the whole expansion with it, and a block entity on layer "0" takes the INSERT's layer. ACadSharp 3.7.1 and acadrust 0.5.5 expose no other per-insert override (source read, not run). Measured: the decoder applies neither effect. |
| `nested-insert` | `included` | `NON_UNIFORM_BLOCK_SCALE` (106) when the composed transform is not similarity | `vc_nested` | Measured: a three-deep chain with two rotations and two scales lands exactly where my hand computation puts it, see the worked example. Depth is bounded by `max_block_depth`. |
| `block-twice-two-scales` | `included` | none | `vc_nested` | Measured: one block inserted at scale 1 and at scale 3 gives two records with different coordinates and the same item handle (`4C`). The handle is not an instance identity, which N4.4 needs to know. |
| `anonymous-block` | `included` | none | `vc_anonymous`, `sample_AC1032` | Measured: an INSERT of `*U1` expands like any block. The decoder gives no block name. ACadSharp exposes `BlockRecord.IsAnonymous` and `Source`. |
| `dynamic-block` | `included` | none | `BLOCKVISIBILITYPARAMETER`, `sample_AC1032` | An INSERT of a dynamic block names the anonymous `*U` block AutoCAD stored for that instance. The decoder draws that stored geometry (20 records for the four instances, equal to what the rule counts). The dynamic definition, its parameters and the link back to the source block are not on the decoder boundary. |
| `xref` | `Unsupported` | `UNRESOLVED_BLOCK` (105) | `g13_xref` | Report as external, see the Xrefs section. Measured: the decoder emits warning 105 naming `../not-resolved/other.dwg` and no geometry. |

## The v1 visible flattened drawing

This is the rule. Given a drawing and a view, the visible flattened drawing is
the multiset of leaf entities, each with its composed transform, that the
following clauses select. `probe/Program.cs rule` applies it to any DWG and
compares the result with what the decoder emitted, so it is a rule a test can
apply to any fixture.

A leaf is an entity that is not an INSERT. A DIMENSION counts as an INSERT of its
anonymous geometry block, placed on the dimension's own layer.

- **V1-1 Space.** Only model space contributes: the view with `ViewKind::Model`.
  A layout view is refused (`Unsupported`), never merged into the drawing and
  never silently skipped.
- **V1-2 Effective layer.** A leaf's effective layer is its own layer, except
  that a leaf on layer "0" inside an INSERT takes the layer of the nearest
  enclosing INSERT.
- **V1-3 Visibility.** An entity is visible only if its effective layer is on
  and thawed. An INSERT that is not visible contributes nothing, whatever its
  block holds. Locked, no-plot and viewport-frozen layers are visible. Colour,
  lineweight and linetype play no part. An entity's own invisible flag (DXF group
  60, `Entity.IsInvisible` in ACadSharp) is a separate switch from layer state,
  and v1 does not read it: the rule probe does not look at it and the decoder
  does not read it either (no `IsInvisible` anywhere in `libviprs-dep/acadsharp/native`,
  source read, not run). So an entity that is invisible on a visible layer is
  drawn by the decoder and counted visible by the rule. No fixture here sets the
  flag, so this is unmeasured, and the rule is not claimed to cover it.
- **V1-4 Instances.** Each visible INSERT contributes its block's visible
  leaves once per instance (a MINSERT is rows times columns instances), under
  the composed transform. Identical leaves reached through two INSERTs appear
  twice, once per instance.
- **V1-5 External and missing.** An INSERT of an xref, or of no block at all,
  contributes nothing and is reported with `UNRESOLVED_BLOCK` (105). It is
  never omitted without that report.

One more clause is about reporting rather than selection: every entity
excluded by V1-3 is counted per layer in the report (`LAYER_HIDDEN`, proposed),
and a drawing with viewport freezes carries one `VIEWPORT_STATE_IGNORED`. A
reader of the report can tell "nothing was on that layer" from "the filter lost
it".

### Worked example: vc_layers

Model space holds four lines `m1`..`m4` on `VC_ON`, `VC_OFF` (off),
`VC_FROZEN` (frozen) and layer "0" (which the file turns off). Block `VC_BLK`
holds `b1` on "0", `b2` on `VC_ON` and `b3` on `VC_OFF`. It is inserted three
times, on `VC_ON` at x=20, on `VC_OFF` at x=30 and on `VC_FROZEN` at x=40.

Applying the rule by hand:

1. `m1` is on `VC_ON`: visible, `(0,0)-(10,0)`. `m2`, `m3` and `m4` are on an
   off, a frozen and an off layer: excluded (V1-3). That is 1 visible, 3 excluded.
2. The INSERT at x=20 is on `VC_ON`: visible. Inside it `b1` is on "0", so its
   effective layer is `VC_ON` (V1-2): visible, `(20,0)-(21,0)`, even though
   layer "0" is off. `b2` is on `VC_ON`: visible, `(20,1)-(21,1)`. `b3` is on
   `VC_OFF`: excluded.
3. The INSERTs at x=30 and x=40 are on `VC_OFF` and `VC_FROZEN`: excluded with
   their whole expansion (V1-3), three leaves each.

Visible: `m1`, `b1`, `b2`, so 3 leaves. Excluded: `m2`, `m3`, `m4`, `b3`, and
the six leaves of the two excluded INSERTs, 10 entities counted. The rule probe
reports `rule_visible_leaves 3` and `rule_hidden_entities 6` (it stops at the
excluded INSERT, so it counts the INSERT, not its three leaves).

The decoder emitted **13** records for model space: all four lines and all
three leaves of all three INSERTs. Ten of them are excluded by the rule. That
gap is the layer limitation in Provider limitations, measured, not argued.

### Worked example: vc_nested

`VC_INNER` is the line `(0,0)-(1,0)`. `VC_MID` holds the line `(0,0)-(0,1)` and an
INSERT of `VC_INNER` at `(1,0)`, rotation 90 degrees, scale 2. `VC_OUTER` holds an
INSERT of `VC_MID` at `(3,0)`, rotation 90 degrees, scale 1. Model space holds
`VC_INNER` at `(10,0)` scale 1, `VC_INNER` at `(20,0)` scale 3, and `VC_OUTER`
at `(100,100)` scale 2. Everything is on `VC_ON`, so V1-3 selects everything and
the example is about V1-4.

Each leaf is scaled, then rotated, then translated, innermost INSERT first:

- `VC_INNER` at scale 1: `(10,0)-(11,0)`.
- `VC_INNER` at scale 3: `(20,0)-(23,0)`. The same block twice at two scales
  is two leaves (V1-4).
- The line in `VC_MID`, under `VC_OUTER`: rotate 90 degrees gives `(0,0)-(-1,0)`,
  translate `(3,0)` gives `(3,0)-(2,0)`, scale 2 gives `(6,0)-(4,0)`, translate
  `(100,100)` gives `(106,100)-(104,100)`.
- `VC_INNER` under `VC_MID` under `VC_OUTER`: scale 2 gives `(0,0)-(2,0)`,
  rotate 90 degrees gives `(0,0)-(0,2)`, translate `(1,0)` gives `(1,0)-(1,2)`
  in `VC_MID`. In `VC_OUTER`: rotate gives `(0,1)-(-2,1)`, translate `(3,0)`
  gives `(3,1)-(1,1)`. In the world: scale 2 gives `(6,2)-(2,2)`, translate gives
  `(106,102)-(102,102)`.

Four leaves, and the decoder emitted exactly these four, with these
coordinates (`vc_nested`, records `4C`, `4C`, `50`, `4C`). The rule probe reports
zero differences. This is the one place the decoder and the rule agree, and it
agrees to the last digit shown.

## Paper space

Paper space in v1: out

The call, and what justifies it:

- A layout of a real AutoCAD drawing (`sample_AC1032`) holds no drawing of
  its own. In `sample_AC1032` the three layouts (`Layout1`, `Layout2`, `MyLayout`) hold two
  entities each, a paper-image viewport and one user viewport, and nothing else.
  What shows in a layout is model space seen through those viewports. On
  `sample_AC1032` the decoder emits 2 warnings per layout (code 100, "VIEWPORT is
  not a primitive this version flattens") and no geometry. So "decode the layout
  view" of `sample_AC1032` produces nothing a tile pipeline could draw. That is
  one AutoCAD drawing, which is as far as this sentence reaches.
- To make a layout mean something, v1 would need the viewport transform, the
  clip, the per-viewport layer freezes, and a decoder that returns model space
  filtered by them. The object models have the inputs (below). The decoder does
  not: no viewport record exists on wire version 2, and layer state does not
  cross at all.
- `src/vector/tiler.rs` already refuses a layout view with
  `TilerError::ViewNotEnabled`. This keeps that behaviour and names the code:
  `VIEW_NOT_ENABLED` (508), whose doc comment already says "until the viewport
  transform and clipping fixtures pass".

The fixtures that would let paper space in are named under "What would let
paper space in".

### Viewport transform evidence

For a plan-view viewport (`ViewDirection = +Z`) the fields give an affine map
from model to paper, with `s = Height / ViewHeight`:

```text
paper = Center + s * R(twist') * (model - ViewCenter)
```

`twist'` is the viewport's twist with a sign I have **not** verified (DXF group
51 is counterclockwise, but I have no AutoCAD-authored fixture with a nonzero
twist to say which way the stored value turns the model).

I have no measurement that checks this map against what AutoCAD draws, so the
document does not offer one. Evaluating it on the `vc_paper` viewports (`4E`:
`Center (80,100)`, `ViewCenter (50,5)`, `Height 80`, `ViewHeight 80`) only sends
`ViewCenter` to `Center`, which is true by construction of the formula for any
`s` and any twist, and the same is true of viewport `4F` (`ViewHeight 160`). Those
points say nothing about whether the scale or the twist is right, so I do not cite
them as evidence. What the call rests on is structural, and measured: a real
layout holds only VIEWPORT entities, and the decoder has no viewport record (see
the first bullet above and L2). The fields themselves come from what ACadSharp
read back from the file, and acadrust agrees with them.

What the real drawings show about those inputs:

| observation | fixture | consequence |
|---|---|---|
| Model-view viewports (`representsPaper=False`, id 2 in each layout) with `Height / ViewHeight` of 0.693488 (two layouts) and 0.696757 (one), twist 0 | `sample_AC1032` | three layouts of one AutoCAD drawing, all untwisted, rectangular and plan-view, so exact. That is one file, not a corpus, and I do not call it the common case. The `Height / ViewHeight` of 1.0 on the id 1 viewport of each layout is the paper-background viewport (`representsPaper=True`), not a model view, and says nothing about scale |
| A model-view viewport (handle `BCE`, id 2) with `ViewDirection = (-1,-1,1)`, `scale` 0.006097 | `AecObjects` | not a 2D affine at all, so it can only be bounded |
| Both model-view viewports (`BCE` and `1040`) are on layer `Defpoints` (a no-plot layer); the paper-background viewport is on layer 0 | `AecObjects` | viewport visibility interacts with layer state |
| `ViewCenter = (0,0)`, `ViewHeight = 0` and `scale = 0` on every viewport, the model-view one (`representsPaper=False`, id 2) included, so `s` is undefined | `sample_AC1014` | the transform cannot be recovered from this provider on AC1014. The same drawing saved as AC1015 reads `ViewHeight 9.228705` and `ViewCenter (6,4.5)` on that same viewport, and so does AC1018 and AC1032. The paper-background viewport (id 1) differs too (`ViewHeight 0` against 8.993), so it is not a confound of that viewport. I cannot tell a reader limitation from R14 storage. Output: the describe section of `results-x64.txt` for these files |
| No viewport in any real fixture has a twist, a viewport freeze or a non-rectangular clip | all real fixtures | those three rest on `vc_paper` alone, which is spike-authored |

Exact versus bounded, as the issue asks:

- **Exact:** untwisted, rectangular, plan-view, with a nonzero `ViewHeight`. The
  clip is the rectangle `Center +/- (Width, Height)/2`.
- **Bounded only:** a twisted viewport (convention unverified), a non-rectangular
  clip (the boundary is an entity reached by handle, here a circle in
  `vc_paper`), a non-plan view direction, a perspective viewport, and any viewport
  whose `ViewHeight` is zero.

Two more things I measured on `vc_paper`, which is spike-authored and not a real
drawing: the clip boundary circle (handle `50`) is also an ordinary drawn entity
in paper space, so on `vc_paper` the decoder does emit it, and a paper-space line
`4D`, as geometry of the layout view (this contrasts with `sample_AC1032`, whose
layouts yield none), and the `FrozenLayers` list survives a write and read through
both ACadSharp and acadrust. Neither exposes per-viewport layer property
overrides (colour, linetype and so on), checked in source and not run.

### What would let paper space in

A known-truth DWG, from the O epic's writer, with: a twisted viewport, a
viewport with a freeze, a non-rectangular clip, and one non-plan view, each with
an `expected.math.json` stating the model-to-paper map. Until that exists, any
v1 paper-space support would rest on my reading of the field meanings.

## Xrefs

Decision: report-as-external

Diagnostic: `UNRESOLVED_BLOCK` (105), carrying the path the drawing recorded and
the item handle of the INSERT.

I rejected the other two and I wrote down why:

- Resolve: rejected for v1. The decoder never opens a second file ("no network,
  no second read", `Flattener.ExpandInsert`), an xref path is attacker-supplied
  input, and a resolved xref brings its own layers, units and base point that
  nothing in this spike measured.
- Reject (fail the whole conversion): rejected, because a drawing with one
  unresolved xref is still mostly a drawing, and a hard failure turns a partial
  result into no result.
- Omit silently: rejected. It is the failure mode this epic exists to avoid. A
  reader of the output cannot tell a drawing with no xref from one whose xref
  was lost.

Measured on `g13_xref`: the decoder emits no geometry and one warning,
`UNRESOLVED_BLOCK` with the message "external reference ../not-resolved/other.dwg
is not resolved, and this decoder never resolves one", handle `4C`. The rule
probe's `rule_external 1` matches it. That is what V1-5 requires.

N4.5 (#1076) owns the xref policy proper. This only fixes the v1 default so N4.5
starts from a decision rather than a blank.

## What the decoder exposes

This is for N4.9 (#1233), which classifies walls and columns from layer and
block rules. The short answer is that it cannot, with the decoder as it is.

What a `Primitive` carries across `CadDecoder` (`src/cad/primitive.rs`): its
geometry, and an `Origin` with an optional `ItemHandle` and a bool for "came
from expanding an insert". That is all. The wire format (`docs/WIRE.md` in
libviprs-dep, wire version 2) carries `handle` and `flags` from the entity's
common fields and nothing else.

| what | on the decoder boundary | in ACadSharp 3.7.1 | in acadrust 0.5.5 |
|---|---|---|---|
| layer visibility (on, frozen) | no. Not applied, not reported | `Layer.IsOn`, `LayerFlags.Frozen`, `FrozenNewViewports`, `Locked` | `Layer::is_off()`, `is_frozen()`, `is_visible()`, `flags.frozen_in_new_viewport` |
| layer name of an entity | no | `Entity.Layer.Name` | `EntityCommon.layer: String` |
| layer table | no | `CadDocument.Layers` | `CadDocument.layers` |
| block name of an INSERT | no, and it is lost on expansion. It appears only inside the text of a warning when the block is unresolved | `Insert.Block.Name` | `Insert.block_name` |
| anonymous, dynamic, xref flags and path | no (xref path only in warning text) | `BlockRecord.IsAnonymous`, `IsDynamic`, `Source`, `BlockEntity.Flags`, `XRefPath` | `BlockRecord.flags.anonymous`, `is_xref`, `xref_path` |
| per-viewport layer freeze | no | `Viewport.FrozenLayers` | `Viewport.frozen_layers` (handles) |
| the INSERT an expanded leaf came from | no. The leaf's own handle repeats once per instance (`52` three times in `vc_layers`, `4C` twice in `vc_nested`) | yes, in the walk | yes, in the walk |
| view name, kind, extents, entity count | yes (`CadView`) | `Layout` | `Layout` |

Measured, not just read: the acadrust probe on `vc_layers` printed per-entity
layer names, the `visible` answer per layer and the INSERT block name for every
entity, and on `sample_AC1032` it printed the same layer states ACadSharp did.
So **acadrust exposes what #1233 needs at the object-model level**, and
ACadSharp does too. What is missing is the decoder boundary in front of
ACadSharp, which is the `CadDecoder` path libviprs uses today.

For #1233 that leaves three ways to get layer and block names to a rule engine:

1. Annotate each geometry record on the decoder boundary with a layer id and,
   for expanded leaves, the chain of INSERT block names (a wire version 3 change
   in libviprs-dep and `acadsharp-rs`).
2. Look a handle up in a side table the provider returns (layer and block name by
   item handle), keeping the wire as it is.
3. Take the acadrust path through the N4.1 provider traits (#1072), where this
   data is already in the object model.

Option 1 or 2 also unblocks N4.6, which needs layer state to filter at all, and
N4.4, which needs the instance chain. I filed the gap as one issue, #1239, rather
than three.

I also noticed one thing that matters for the classification rules: layer "0" is
meaningless as a classifier inside a block, because of V1-2. A rule engine that
matches the leaf's own layer would misclassify every leaf of a block drawn on "0",
and the rule engine has to use the effective layer.

## Provider limitations

These are cases the decoder cannot expose or apply, so they are recorded as
provider limitations and not as libviprs policy. Provider versions for all of
them: ACadSharp 3.7.1 (libviprs-dep patches applied), libviprs-dep shim
`3.7.1-viprs.2` at commit `0d475bf`, wire version 2, JIT run on native x86_64.
Where acadrust is named it is 0.5.5 at `d35691f`.

| id | limitation | cases it blocks | evidence |
|---|---|---|---|
| L1 | Layer state is neither applied nor exposed by the decoder. ACadSharp has it. | `layer-off`, `layer-frozen`, `insert-layer-override` | `vc_layers`: decoder emitted 13 records, the rule selects 3. Source: the adapter never reads a layer (no `Layer`, `IsOn` or `Frozen` in `libviprs-dep/acadsharp/native`, except the probe fixture writer). |
| L2 | No viewport record exists, and `VIEWPORT` is refused as an unsupported entity. ACadSharp has every viewport field and the frozen-layer list. | `paper-space-layout`, `viewport-clip-exact`, `viewport-clip-bounded`, `layer-frozen-in-viewport` | `sample_AC1032`: 6 `VIEWPORT` warnings (code 100) across 3 layout views, no geometry. |
| L3 | No layer name, block name or insert chain on the wire. | `anonymous-block`, `dynamic-block`, and #1233 | `vc_anonymous`: the records for `*U1` carry only handle `4D` and flags 1. |
| L4 | A leaf's handle is the same for every instance. | `block-twice-two-scales` | `vc_nested`: handle `4C` twice, with different coordinates. |
| L5 | Dynamic-block definition, parameters and the anonymous-to-source link are not exposed. ACadSharp has `BlockRecord.Source`. | `dynamic-block` | `BLOCKVISIBILITYPARAMETER`: 4 instances draw the stored anonymous geometry, 20 records, rule 20. The decoder cannot say what state they are in. |
| L6 | An INSERT inside a DIMENSION's geometry block (arrowheads, tick blocks) is skipped with no warning. This is not a view-context case, but it breaks the "nothing is silently omitted" principle the xref decision rests on. | none of the table's rows, but it moves the numbers | `Flattener.DimensionBody` `continue`s on `Insert`. On `sample_AC1032` the rule counts 146 leaves and the decoder emitted 134, and the 12 missing are all in `_ArchTick` and `_BoxBlank` instances (6 handles, 2 instances each). |
| L7 | AC1014 reads every viewport, the model-view ones (id 2) included, with `ViewCenter (0,0)`, `ViewHeight 0` and `scale 0`. | `viewport-clip-exact` on R14 | `sample_AC1014` against `sample_AC1015`, viewport handles `245`, `256` and `26B`, in the describe section of `results-x64.txt`. Not measured: whether R14 stores the fields at all. |

Cases the real corpus could not exercise, so the evidence is spike-authored
only: an entity on an off or frozen layer (`sample_AC1032` has `Layer_Off` and
`Layer_Freeze` defined and uses neither in model space, the rule probe reports
zero hidden entities on every real fixture), a viewport freeze, a twisted or
non-rectangular viewport, an xref (only a writer-authored one exists) and a
per-insert override.

## Where the numbers in this document come from

| what | file |
|---|---|
| every probe run cited above | `docs/spikes/view-context/results-x64.txt` |
| the .NET probe (`author`, `describe`, `shim`, `rule`) | `docs/spikes/view-context/probe/` |
| the acadrust probe | `docs/spikes/view-context/acadrust-probe/` |
| staging, run and results scripts | `docs/spikes/view-context/stage.sh`, `run.sh`, `results.sh` |
| the four spike-authored fixtures | `docs/spikes/view-context/fixtures/` |
| the checks that hold this document to the issue's criteria | `tests/view_context_spike_doc.rs` |

The `rule` mode is written against ACadSharp's object model and never against
the decoder, so a disagreement is a finding about the decoder and not a program
disagreeing with itself. It compares one-record kinds only (line, arc, circle,
ellipse, lightweight polyline, spline), because a hatch, mesh or mline fans out
into several records under one handle and a point or xline is refused, and
counting those would measure entity coverage rather than layer handling.
