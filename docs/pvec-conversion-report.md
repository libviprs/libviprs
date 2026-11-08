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
| `capabilities` | The capability profile the conversion ran under (N7.2). Optional and ignorable, so it is not a version bump: absent when none was recorded. |
| `profile` | `Strict`, `Default`, `Permissive` or `Custom`. |
| `policy` | `RejectInput`, `RejectEntity` or `Downgrade`. |
| `exact_only` | True when only `SupportedExact` entities passed. |
| `allowed` | The allowed PVec capability ids, ascending. |
| `georeference` | The file name of the georeference sidecar (N7.3) published beside the artifact, `<artifact>.georef.json`. Optional and ignorable, so it is not a version bump: absent when the conversion wrote none. |
| `diagnostics` | The diagnostics of the conversion, in entity id order (N7.5), each as described in `docs/pvec-diagnostics.md`. Optional and ignorable, so it is not a version bump: absent when the conversion made none. |
| `components` | What the component streams of a conversion did (`PvecConversion::components`, issue #1525). Optional and ignorable, so it is not a version bump: absent when the conversion wrote no stream. `streams` (the streams written, in file order: `layer`, `line-style`, `provenance`), `dropped` (rows PVec cannot hold, keyed `stream.reason`, for example `layer.name-too-long`, with the count) and `colours_ignored` (DWG colour indices dropped under the monochrome profile, counted only when the line style stream is selected). |
| `regions` | What the room and corridor step found (`PvecConversion::regions`, issue #1526). Optional and ignorable, so it is not a version bump: absent when the step was not asked for. `rooms`, `corridors`, `gaps_closed` and `unlabelled` (faces no rule could label), `labelled` (regions with a TEXT inside), `written` (regions in PVec's semantic class stream), `dropped` (regions left out of that stream, keyed by reason: `derived-boundary` for a room that runs along a door-gap segment, which is never written as geometry, then `unlabelled`, `unwritten-boundary`, `no-boundary`, `ring-too-long`) and `refused` (a reason code such as `regions.node-limit` when the plan is past a limit; the conversion carries on without regions). The walls are the straight lines the default element rules call walls, and the gap tolerance is in drawing units. |
| `pdf` | What a PDF page conversion did beside its vector entities (`PdfConversion`, issue #1491). Optional and ignorable, so it is not a version bump: absent for a DWG. The coordinates of a PDF conversion are the page's own, in points, unscaled (the manifest says `metres_per_unit` 1.0 because PVec cannot declare points). `page`, `images`, `rasters` (images written), `images_refused` (reason key to count, keys in `docs/pvec-raster-producer.md`), `image_notes` (note key to count), `fills` (fill policy reason to count, `docs/pvec-hatch-policy.md`), `text_omitted`, and `calibration` (issue #1547, optional and ignorable too): `viewports` (the page's `/VP` count), `lgidicts` (its `/LGIDict` count, issue #1548), `viewport` (the index of the one whose `/Measure` was used, absent when none was), `scale` (`declared-unit` when the scale is the point, 0.0254 / 72 m times `/UserUnit`, or `plan-scale` from an `/RL` measure), `georeference` (the CRS id of a GeoPDF georeference, or `definition` when only its WKT names it, absent when there is none) and `notes`, why something the page said was not used: `rl-unit-unknown`, `rl-anisotropic`, `rl-refused`, `geo-projected-without-definition`, `geo-no-crs`, `geo-refused`, `measure-unread`, `user-unit-ignored`, and for an LGIDict `lgidict-ignored` (a viewport already gave the georeference), `lgidict-not-mapped` (a projection other than UTM on WGS 84, which would need a table), `lgidict-first-of-several`, `lgidict-refused`. An LGIDict that maps is a declared affine into `EPSG:326zz` or `EPSG:327zz` with no control points. |
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
