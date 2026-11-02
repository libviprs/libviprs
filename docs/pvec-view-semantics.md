# What "the drawing" means: view and visibility

Version 1 draws one thing: the visible, flattened model space. The rule is `ViewProfile::default_profile()`, constructed
explicitly so a caller can see what they accepted, and a test pins every value
(`docs/adr/pvec-view-profile.md` has the decision table and its evidence).

<!-- view:begin -->
| field | value | N1.4 row |
|---|---|---|
| `space` | model | model-space |
| `viewport_clipping` | unsupported | viewport-clip-exact |
| `layers.off` | exclude | layer-off |
| `layers.frozen` | exclude | layer-frozen |
| `layers.frozen_in_viewport` | ignore-and-report | layer-frozen-in-viewport |
| `layers.insert_override` | inherit-on-layer-zero | insert-layer-override |
| `nested_inserts.max_depth` | 64 | nested-insert |
| `xrefs` | report-external | xref |
<!-- view:end -->

## Rules in words

- **Layers.** A leaf on an off or a frozen layer is excluded, with the layer and the profile field named in the report.
  A layer frozen only in a viewport is ignored and reported, because v1 has no viewport.
- **Layer zero in a block.** A leaf or INSERT on layer `0` inside an INSERT takes that INSERT's layer.
- **Nesting.** INSERTs are expanded up to `nested_inserts.max_depth`. An INSERT past it is excluded with the reason
  `nested_inserts.max_depth`, and a block that refers to itself is refused by the flattener's source limits
  (`docs/pvec-limits.md`).
- **Space.** Model space only. Asking for a paper layout is refused with `PaperSpaceUnsupported` (VIEW_NOT_ENABLED) when the
  layout exists, or `LayoutNotFound` when it does not. Viewport clipping is `unsupported`.
- **Xrefs.** An external reference is reported, not resolved and not flattened.
- **Limits.** The walk charges the profile's source limits as it goes, so a drawing that fans out is refused with the
  limit named instead of being expanded until memory runs out (#1439).

Every leaf is either kept or excluded with a reason; the report's counts add up to the entities met.
