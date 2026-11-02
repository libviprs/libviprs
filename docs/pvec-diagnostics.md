# Conversion diagnostics

`libviprs::pvec::diagnostics` (issue #1416, N7.5). A `Diagnostic` says what happened to one entity
or to the run: the source, the layout, the entity's handle and class, the block path it was reached
through, its layer, the stage, a severity, a stable code and a stable message id. They stream to an
observer as they are made (`EngineEvent::ConversionDiagnostic`) and the durable copy is the
`diagnostics` section of the conversion report (`docs/pvec-conversion-report.md`), its own section
and never mixed into the geometry.

The code, the severity and the message id are stable. The `message` is for people and can change.
The severity is a property of the code (`DiagnosticCode::severity`), not something read out of the
text.

A supported entity makes no diagnostic, so a drawing that converts cleanly makes none, and with no
observer attached its per-entity path allocates nothing for them.

## The codes

The table is closed. `tests/pvec_1416_diagnostics.rs` reads it from here, and fails on a code that
nothing emits and on one that is emitted and not written down.

| code | meaning | severity | message id | fixture |
|---|---|---|---|---|
| `ENTITY_REQUIRES_CAPABILITY` | An entity needs a PVec capability that is not allowed or not available. | warning | the reason code, `capability-not-allowed` | `an_entity_outside_the_allowed_set` |
| `ENTITY_UNSUPPORTED` | An entity has no mapping into PVec, or PVec refused its geometry. | warning | the reason code, `unmapped-source-class` | `an_entity_the_provider_does_not_map` |
| `ENTITY_IGNORED_BY_POLICY` | The view profile left an entity out. | info | the profile field, `layers.off` | `a_leaf_on_a_hidden_layer` |
| `INSERT_NOT_EXPANDED` | An INSERT was not expanded: it has no handle, is an xref, or has a normal other than Z. | warning | the reason code, `unresolved-block` | `an_insert_without_a_handle` |
| `SOURCE_OPEN_FAILED` | The source could not be opened. | error | `open` | `a_missing_source` |
| `VIEW_FAILED` | The view profile could not be applied. | error | `view` | `a_paper_layout_in_v1` |
| `BLOCKS_FAILED` | Block expansion failed or a limit was exceeded. | error | `limit`, `block-recursion`, `entities-per-block`, `compose`, `provider` | `a_block_depth_limit` |
| `UNIT_REQUIRED` | No unit was given and the provider reports none. | error | `unit-required` | `a_generic_provider_without_a_unit` |
| `CAPABILITY_REJECTED` | An entity needs a capability the run does not allow, and the policy is to stop. | error | `capability-not-allowed` | `reject_input` |
| `WRITE_FAILED` | PVec's writer refused the artifact. | error | `write` | `a_unit_pvec_cannot_declare` |
| `REPORT_FAILED` | The report could not be built, or its counts disagree with the entities met. | error | `report` | `a_report_that_does_not_add_up` |
| `PUBLISH_FAILED` | The artifact and report could not be published. | error | `publish` | `a_destination_that_is_a_directory` |
| `GEOREFERENCE_FAILED` | The georeference sidecar could not be built or written. | error | `georeference` | `a_control_point_that_is_not_finite` |
| `CANCELLED` | The caller cancelled the run. | info | `cancelled` | `a_precancelled_run` |

## Fields

| field | meaning |
|---|---|
| `code` | One of the codes above. |
| `severity` | `Info`, `Warning` or `Error`, from the code. |
| `stage` | `Open`, `View`, `Flatten`, `Classify`, `Write`, `Report` or `Publish`, from the code. |
| `message_id` | The stable identifier under the code. |
| `message` | The wording for people. Not stable. |
| `source` | The source file, when the conversion has a path for it. |
| `layout` | The layout the entity belongs to. |
| `handle` | The entity's handle, when the drawing gave one. |
| `class` | The drawing's class name for the entity. |
| `block_path` | The INSERT chain the entity was reached through, spelled `block@HANDLE/...`. |
| `layer` | The entity's own layer. |
