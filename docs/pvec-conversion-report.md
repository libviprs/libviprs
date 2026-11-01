# The conversion report

Every conversion to PVec emits a machine-readable report of what became of each entity it met
(issue #1093, N6.7). The point is the quiet failure: a drawing converts, looks plausible, and
nothing says that 400 entities were dropped as unsupported or filtered by policy. The code is
`src/pvec/report.rs`; the report is published beside the artifact as `<artifact>.report.json` by
`pvec::publish`, both or neither.

The identity that holds in every report: the rows are exactly the entities encountered, and the
six fidelity counts add up to `encountered`. `ReportBuilder::finish` refuses to build a report
otherwise, so an entity cannot be absent from it.

## Schema, version 1

The JSON is pretty-printed from structs and sorted maps only, so the same input gives the same
bytes, whatever order the rows were pushed in and however many workers produced them.

| field | meaning |
|---|---|
| `schema_version` | The version of this schema, `1`. A reader refuses any other. |
| `totals` | Aggregates over all the rows. |
| `encountered` | Entities the conversion met. Equal to the number of `entities` rows. |
| `by_fidelity` | Rows per fidelity class. All six classes are present, with zeros: `SupportedExact`, `SupportedEquivalent`, `RequiresPVecCapability`, `RequiresExtension`, `Unsupported`, `IgnoredByPolicy`. |
| `by_source_class` | Rows per drawing class name (`LINE`, `SPLINE`...). |
| `by_semantic_type` | Rows per PVec semantic type, for the entities that produced geometry. |
| `view` | The view decisions (N4.6). |
| `space` | The space that was drawn. |
| `included` | Leaves kept. |
| `excluded_by_field` | Leaves excluded, by the view-profile field responsible. |
| `excluded_by_layer` | Leaves excluded, by layer name. |
| `xrefs` | External references (N4.5). |
| `unresolved` | Xrefs that were left out, in the order they were met. |
| `resolved` | Names of the xrefs that were resolved, sorted. |
| `name` | An unresolved xref's name. |
| `path` | The path as the drawing attached it. |
| `resolved_path` | The path under the root, when the attachment was acceptable, else null. |
| `path_refusal` | Why the path was refused (`Absolute`, `EscapesRoot`...), else null. |
| `bounds` | How the entity boxes were obtained (N6.4). |
| `exact` | Boxes in closed form, tight. |
| `hull_upper_bound` | Boxes that are the control hull: enclosing, not tight. |
| `entities` | One row per entity, in id order. |
| `id` | The entity's id from the ordering pass (`pvec::EntityId`). |
| `source_class` | The drawing's class name for the entity. |
| `source_handle` | The drawing's handle for the entity, null when it had none. |
| `semantic_type` | The PVec semantic type, null for an entity that produced no geometry. |
| `fidelity` | The entity's fidelity class, one of the six above. |
| `reason` | The stable reason code (`ReasonCode::code`), present exactly for `RequiresPVecCapability`, `RequiresExtension` and `Unsupported`. |
| `policy_field` | The view-profile field that excluded the entity, present exactly for `IgnoredByPolicy`. |
| `warnings` | Warnings recorded against the entity. |

A row with a refusing class and no `reason`, an `IgnoredByPolicy` row with no `policy_field`, a
supported class with no `semantic_type`, or an unsupported, ignored or capability-less one that has
a type, cannot be built: each ignored or filtered entity names
exactly one reason.

The issue talks about five buckets (exact, equivalent, extension, unsupported, ignored). The code
has six classes, because `RequiresPVecCapability` (PVec has no entry point yet) is told apart from
`Unsupported` (the construct has no mapping). A reader who wants five adds the two.

## Compatibility rule

`schema_version` moves whenever a field is added, renamed, removed or changes meaning, with one
exception: a new optional field that a reader may ignore may be added without a bump, and this
document says so next to it. `tests/pvec_1093_report.rs` pins the full list of field paths next to
the version, so a change cannot slip in without somebody editing both. A report whose
`schema_version` is not the one the reader knows is refused (`ReportError::Schema`).

## What is not in the report

Corpus-level regression reporting (N7.8) and benchmark reports (N8.8) are separate. Preview
coverage in Tier 0 is display fidelity and, when reported, never carries a geometric-exactness
class.
